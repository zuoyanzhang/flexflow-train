#include "task-spec/ops/impl/element_binary.h"
#include "kernels/element_binary_kernels.h"
#include "op-attrs/tensor_dims.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"
#include <cstdlib>
#include <iostream>

namespace FlexFlow {

using namespace FlexFlow::Kernels::ElementBinary;

static bool should_log_element_binary() {
  return std::getenv("FLEXFLOW_DEBUG_ELEMENT_BINARY") != nullptr;
}

static void log_element_binary_forward(ElementBinaryAttrs const &attrs,
                                       GenericTensorAccessorR const &lhs,
                                       GenericTensorAccessorR const &rhs,
                                       GenericTensorAccessorW const &output) {
  std::cerr << "[element-binary] forward op=" << attrs.type
            << " broadcast_lhs=" << attrs.should_broadcast_lhs
            << " broadcast_rhs=" << attrs.should_broadcast_rhs
            << " lhs_shape=" << lhs.shape << " rhs_shape=" << rhs.shape
            << " out_shape=" << output.shape << " lhs_device="
            << lhs.device_type << " rhs_device=" << rhs.device_type
            << " out_device=" << output.device_type << " lhs_ptr=" << lhs.ptr
            << " rhs_ptr=" << rhs.ptr << " out_ptr=" << output.ptr << "\n";
}

static void log_element_binary_backward(ElementBinaryAttrs const &attrs,
                                        GenericTensorAccessorR const &out_grad,
                                        GenericTensorAccessorR const &lhs,
                                        GenericTensorAccessorR const &rhs,
                                        GenericTensorAccessorW const &lhs_grad,
                                        GenericTensorAccessorW const &rhs_grad) {
  std::cerr << "[element-binary] backward op=" << attrs.type
            << " broadcast_lhs=" << attrs.should_broadcast_lhs
            << " broadcast_rhs=" << attrs.should_broadcast_rhs
            << " out_grad_shape=" << out_grad.shape
            << " lhs_shape=" << lhs.shape << " rhs_shape=" << rhs.shape
            << " lhs_grad_shape=" << lhs_grad.shape
            << " rhs_grad_shape=" << rhs_grad.shape
            << " out_grad_device=" << out_grad.device_type
            << " lhs_device=" << lhs.device_type
            << " rhs_device=" << rhs.device_type
            << " lhs_grad_device=" << lhs_grad.device_type
            << " rhs_grad_device=" << rhs_grad.device_type
            << " out_grad_ptr=" << out_grad.ptr << " lhs_ptr=" << lhs.ptr
            << " rhs_ptr=" << rhs.ptr << " lhs_grad_ptr=" << lhs_grad.ptr
            << " rhs_grad_ptr=" << rhs_grad.ptr << "\n";
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input_lhs = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();

  std::optional<ElementBinaryPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.type,
                  attrs.should_broadcast_lhs,
                  attrs.should_broadcast_rhs,
                  input_lhs.shape,
                  input_rhs.shape,
                  output.shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementBinaryPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_element_binary().value();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  device_handle_t handle = acc.get_ff_handle();

  auto input_lhs = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  if (should_log_element_binary()) {
    log_element_binary_forward(attrs, input_lhs, input_rhs, output);
  }

  size_t output_num_elements = static_cast<size_t>(
      get_num_elements(output.shape.dims).int_from_positive_int());

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementBinary] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input_lhs.get_float_ptr(),
                 input_rhs.get_float_ptr(),
                 output.get_float_ptr(),
                 output_num_elements,
                 attrs.type,
                 attrs.should_broadcast_lhs,
                 attrs.should_broadcast_rhs,
                 handle);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementBinaryPerDeviceState per_device_state =
      acc.get_per_device_op_state().require_element_binary().value();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  device_handle_t handle = acc.get_ff_handle();

  auto input_lhs = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);

  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto input_lhs_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::LHS_INPUT);
  auto input_rhs_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::RHS_INPUT);

  if (should_log_element_binary()) {
    log_element_binary_backward(attrs,
                                output_grad,
                                input_lhs,
                                input_rhs,
                                input_lhs_grad,
                                input_rhs_grad);
  }

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementBinary] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_lhs.get_float_ptr(),
                 input_rhs.get_float_ptr(),
                 input_lhs_grad.get_float_ptr(),
                 input_rhs_grad.get_float_ptr(),
                 attrs.type,
                 attrs.should_broadcast_lhs,
                 attrs.should_broadcast_rhs,
                 handle);
}

TaskImplFunction get_element_binary_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_element_binary_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_element_binary_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow
