#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/optimizer_attrs.h"
#include "pcg/optimizer_slot_name.dtg.h"
#include "task-spec/fwb_tensor_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/map_keys_and_values.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include "utils/overload.h"
#include <cstring>
#include <optional>

namespace FlexFlow {

namespace {

bool slot_has_fwb_role(DynamicTensorSlot const &slot, FwbTensorType role) {
  if (!slot.slot_tensor_role.has_value()) {
    return false;
  }
  std::optional<FwbTensorType> slot_role =
      slot.slot_tensor_role.value().try_require_fwb_tensor();
  return slot_role.has_value() && slot_role.value() == role;
}

std::optional<GenericTensorAccessorR> find_first_read_accessor_for_role(
    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> const &values,
    FwbTensorType role) {
  for (auto const &[slot, value] : values) {
    if (!slot_has_fwb_role(slot, role) || !value.accessor.has_value()) {
      continue;
    }
    DynamicTensorAccessor accessor = value.accessor.value();
    if (accessor.is_read()) {
      return accessor.require_read();
    }
    return read_only_accessor_from_write_accessor(accessor.require_write());
  }
  return std::nullopt;
}

std::optional<GenericTensorAccessorW> find_first_write_accessor_for_role(
    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> const &values,
    FwbTensorType role) {
  for (auto const &[slot, value] : values) {
    if (!slot_has_fwb_role(slot, role) || !value.accessor.has_value()) {
      continue;
    }
    DynamicTensorAccessor accessor = value.accessor.value();
    if (accessor.is_write()) {
      return accessor.require_write();
    }
  }
  return std::nullopt;
}

size_t get_accessor_size_in_bytes(TensorShape const &shape) {
  return get_size_in_bytes(shape).unwrap_num_bytes().unwrap_nonnegative();
}

void zero_accessor(GenericTensorAccessorW const &dst) {
  size_t num_bytes = get_accessor_size_in_bytes(dst.shape);
  if (num_bytes == 0) {
    return;
  }

  if (dst.device_type == DeviceType::CPU) {
    std::memset(dst.ptr, 0, num_bytes);
  } else {
    checkCUDA(cudaMemset(dst.ptr, 0, num_bytes));
  }
}

void copy_or_zero_accessor(GenericTensorAccessorW const &dst,
                           GenericTensorAccessorR const &src) {
  size_t dst_num_bytes = get_accessor_size_in_bytes(dst.shape);
  size_t src_num_bytes = get_accessor_size_in_bytes(src.shape);
  if (dst_num_bytes == 0) {
    return;
  }

  if (src_num_bytes < dst_num_bytes) {
    zero_accessor(dst);
    return;
  }

  if (src.ptr == dst.ptr && src.device_type == dst.device_type) {
    return;
  }
  copy_accessor_data_to_l_from_r(dst, src);
}

void execute_parallel_passthrough(DynamicNodeInvocation const &invocation,
                                  FwbTensorType role) {
  std::optional<GenericTensorAccessorR> src =
      find_first_read_accessor_for_role(invocation.inputs, role);
  std::optional<GenericTensorAccessorW> dst =
      find_first_write_accessor_for_role(invocation.outputs, role);

  if (!dst.has_value()) {
    return;
  }
  if (!src.has_value()) {
    zero_accessor(dst.value());
    return;
  }

  copy_or_zero_accessor(dst.value(), src.value());
}

bool invocation_is_parallel_op(DynamicNodeInvocation const &invocation) {
  if (!invocation.node_attrs.op_attrs.has_value()) {
    return false;
  }
  std::optional<PCGOperatorAttrs> pcg_attrs =
      invocation.node_attrs.op_attrs.value().try_require_pcg_op();
  return pcg_attrs.has_value() && is_parallel_op(pcg_attrs.value());
}

} // namespace

TaskTensorParameter make_task_tensor_parameter_from_dynamic_slot(
    DynamicTensorSlot const &slot,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {
  return assert_unwrap(slot.slot_tensor_role)
      .visit<TaskTensorParameter>(overload{
          [&](FwbTensorType const &fwb_tensor) {
            switch (fwb_tensor) {
              case FwbTensorType::FORWARD:
                return make_task_tensor_parameter_fwd(slot.slot_name);
              case FwbTensorType::GRADIENT:
                return make_task_tensor_parameter_grad(slot.slot_name);
              default:
                PANIC("Unhandled FwbTensorType", fmt::to_string(fwb_tensor));
            }
          },
          [&](DynamicOptimizerTensorRole const &optimizer_tensor) {
            return make_task_tensor_parameter_opt(
                slot.slot_name, optimizer_tensor.optimizer_slot_name);
          },
          [&](DynamicLossTensorRole const &loss_tensor) {
            return make_task_tensor_parameter_loss();
          },
      });
}

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx) {
  auto make_param = [&](DynamicTensorSlot const &slot) {
    return make_task_tensor_parameter_from_dynamic_slot(slot, optimizer_attrs);
  };
  auto get_accessor = [](DynamicValueAttrs const &value) {
    return assert_unwrap(value.accessor);
  };
  std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
      tensor_slots_backing = binary_merge_disjoint_maps(
          map_keys_and_values(invocation.inputs, make_param, get_accessor),
          map_keys_and_values(invocation.outputs, make_param, get_accessor));

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      /*allocator=*/allocator,
      /*tensor_slots_backing=*/tensor_slots_backing,
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*op_attrs=*/
      and_then(invocation.node_attrs.op_attrs,
               [](TrainingOperationAttrs const &op_attrs) {
                 return op_attrs.try_require_pcg_op();
               }),
      /*loss_attrs=*/
      and_then(invocation.node_attrs.op_attrs,
               [](TrainingOperationAttrs const &op_attrs) {
                 return op_attrs.try_require_loss();
               }),
      /*per_device_op_state=*/per_device_op_state,
      /*optimizer_attrs=*/optimizer_attrs,
      /*device_idx=*/device_idx);
}

std::optional<milliseconds_t> execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx) {
  if (invocation_is_parallel_op(invocation)) {
    DynamicTaskType task_type = assert_unwrap(invocation.node_attrs.task_type);
    if (task_type == DynamicTaskType::FWD) {
      execute_parallel_passthrough(invocation, FwbTensorType::FORWARD);
    } else if (task_type == DynamicTaskType::BWD) {
      execute_parallel_passthrough(invocation, FwbTensorType::GRADIENT);
    } else {
      PANIC("Unhandled parallel op task type", task_type);
    }
    return std::nullopt;
  }

  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/invocation,
          /*allocator=*/allocator,
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/ff_handle,
          /*per_device_op_state=*/per_device_op_state,
          /*optimizer_attrs=*/optimizer_attrs,
          /*device_idx=*/device_idx);

  DynamicTaskType task_type = assert_unwrap(invocation.node_attrs.task_type);
  std::optional<milliseconds_t> result;
  switch (task_type) {
    case DynamicTaskType::FWD: {
      ComputationGraphOpAttrs op_attrs =
          assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
              assert_unwrap(invocation.node_attrs.op_attrs).require_pcg_op()));
      result = call_fwd_task_impl(op_attrs, arg_accessor);
    } break;
    case DynamicTaskType::BWD: {
      ComputationGraphOpAttrs op_attrs =
          assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
              assert_unwrap(invocation.node_attrs.op_attrs).require_pcg_op()));
      result = call_bwd_task_impl(op_attrs, arg_accessor);
    } break;
    case DynamicTaskType::UPD:
      call_update_task_impl(assert_unwrap(optimizer_attrs), arg_accessor);
      break;
    case DynamicTaskType::LOSS:
      call_loss_task_impl(arg_accessor);
      break;
    default:
      PANIC("Unhandled DynamicTaskType", task_type);
  }
  return result;
}

} // namespace FlexFlow
