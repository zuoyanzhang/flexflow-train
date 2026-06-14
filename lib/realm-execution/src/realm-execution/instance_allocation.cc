#include "realm-execution/instance_allocation.h"
#include "local-execution/tensor_allocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/make.h"
#include "utils/containers/map_values.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

static double bytes_to_mib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static size_t get_allocation_size_bytes(DynamicValueAttrs const &value) {
  TensorShape shape = get_piece_shape(value.parallel_tensor_shape.value());
  return get_size_in_bytes(shape)
      .unwrap_num_bytes()
      .size_t_from_nonnegative_int();
}

struct AllocationPoolKey {
  MachineSpaceCoordinate device_coord;
  TensorShape shape;

  bool operator==(AllocationPoolKey const &other) const {
    return this->device_coord == other.device_coord &&
           this->shape == other.shape;
  }
};

struct AllocationPoolKeyHash {
  size_t operator()(AllocationPoolKey const &key) const {
    size_t result = std::hash<MachineSpaceCoordinate>{}(key.device_coord);
    result ^= std::hash<TensorShape>{}(key.shape) + 0x9e3779b9 +
              (result << 6) + (result >> 2);
    return result;
  }
};

struct ValueAllocationInfo {
  size_t first_use = std::numeric_limits<size_t>::max();
  size_t last_use = 0;
  std::optional<MachineSpaceCoordinate> device_coord = std::nullopt;
};

struct ReusableAllocation {
  std::pair<Realm::RegionInstance, Realm::Event> instance;
  size_t last_use = 0;
};

static MachineSpaceCoordinate
    get_allocation_device_coord(DynamicNodeAttrs const &node_attrs,
                                DynamicValueAttrs const &value) {
  if (value.mapping.has_value() && value.shard_coord.has_value()) {
    return value.mapping.value().at_l(value.shard_coord.value());
  }

  return assert_unwrap(node_attrs.device_coord);
}

static void print_instance_allocation_summary(
    std::vector<DynamicNodeInvocation> const &execution_order,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated) {
  std::unordered_set<DynamicValueAttrs> seen_values;
  std::unordered_map<MachineSpaceCoordinate, size_t> bytes_by_device;
  size_t total_bytes = 0;

  auto visit_value = [&](DynamicNodeAttrs const &node_attrs,
                         DynamicValueAttrs const &value) {
    if (contains_key(preallocated, value) ||
        seen_values.find(value) != seen_values.end()) {
      return;
    }

    seen_values.insert(value);
    MachineSpaceCoordinate device_coord =
        get_allocation_device_coord(node_attrs, value);
    size_t bytes = get_allocation_size_bytes(value);
    total_bytes += bytes;
    bytes_by_device[device_coord] += bytes;
  };

  for (DynamicNodeInvocation const &invocation : execution_order) {
    for (DynamicValueAttrs const &input : values(invocation.inputs)) {
      visit_value(invocation.node_attrs, input);
    }
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      visit_value(invocation.node_attrs, output);
    }
  }

  std::cerr << "[instance-allocation] invocations=" << execution_order.size()
            << ", tensor_instances=" << seen_values.size()
            << ", estimated_total_mib=" << bytes_to_mib(total_bytes) << "\n";
  for (auto const &[device_coord, bytes] : bytes_by_device) {
    std::cerr << "[instance-allocation] device=" << device_coord
              << ", estimated_mib=" << bytes_to_mib(bytes) << "\n";
  }
}

static void print_reused_instance_allocation_summary(
    TensorInstanceBacking const &backing,
    std::unordered_map<DynamicValueAttrs, ValueAllocationInfo> const &infos) {
  std::unordered_set<Realm::RegionInstance::id_t> seen_instances;
  std::unordered_map<MachineSpaceCoordinate, size_t> bytes_by_device;
  size_t total_bytes = 0;

  for (auto const &[value, instance_and_ready] : backing.backing) {
    Realm::RegionInstance const &instance = instance_and_ready.first;
    if (!seen_instances.insert(instance.id).second) {
      continue;
    }
    MachineSpaceCoordinate device_coord =
        assert_unwrap(infos.at(value).device_coord);
    size_t bytes = get_allocation_size_bytes(value);
    total_bytes += bytes;
    bytes_by_device[device_coord] += bytes;
  }

  std::cerr << "[instance-allocation] reused_tensor_instances="
            << seen_instances.size()
            << ", estimated_reused_total_mib=" << bytes_to_mib(total_bytes)
            << "\n";
  for (auto const &[device_coord, bytes] : bytes_by_device) {
    std::cerr << "[instance-allocation] reused device=" << device_coord
              << ", estimated_mib=" << bytes_to_mib(bytes) << "\n";
  }
}

static void record_value_use(
    DynamicNodeAttrs const &node_attrs,
    DynamicValueAttrs const &value,
    size_t invocation_idx,
    std::unordered_map<DynamicValueAttrs, ValueAllocationInfo> &infos) {
  ValueAllocationInfo &info = infos[value];
  info.first_use = std::min(info.first_use, invocation_idx);
  info.last_use = std::max(info.last_use, invocation_idx);
  if (!info.device_coord.has_value()) {
    info.device_coord = get_allocation_device_coord(node_attrs, value);
  }
}

static std::unordered_map<DynamicValueAttrs, ValueAllocationInfo>
    get_value_allocation_infos(
        std::vector<DynamicNodeInvocation> const &execution_order) {
  std::unordered_map<DynamicValueAttrs, ValueAllocationInfo> result;
  for (size_t i = 0; i < execution_order.size(); i++) {
    DynamicNodeInvocation const &invocation = execution_order.at(i);
    for (DynamicValueAttrs const &input : values(invocation.inputs)) {
      record_value_use(invocation.node_attrs, input, i, result);
    }
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      record_value_use(invocation.node_attrs, output, i, result);
    }
  }
  return result;
}

std::pair<Realm::RegionInstance, Realm::Event>
perform_instance_allocation_for_value(
    MachineSpaceCoordinate const &device_coord, DynamicValueAttrs const &value,
    RealmContext &ctx) {
  ASSERT(value.accessor == std::nullopt);

  TensorShape shape = get_piece_shape(value.parallel_tensor_shape.value());

  Realm::Processor proc = ctx.map_device_coord_to_processor(device_coord);
  Realm::Memory memory = ctx.get_nearest_memory(proc);
  return ctx.create_instance(memory, shape, Realm::ProfilingRequestSet());
}

TensorInstanceBacking perform_instance_allocation(
    std::vector<DynamicNodeInvocation> const &execution_order,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    RealmContext &ctx) {
  for (DynamicValueAttrs const &v : keys(preallocated)) {
    ASSERT(v.accessor == std::nullopt);
  }

  print_instance_allocation_summary(execution_order, preallocated);

  TensorInstanceBacking result = make_empty_tensor_instance_backing();

  std::unordered_map<DynamicValueAttrs, ValueAllocationInfo> infos =
      get_value_allocation_infos(execution_order);
  std::unordered_set<DynamicValueAttrs> value_key_set = keys(infos);
  std::vector<DynamicValueAttrs> values_to_allocate(value_key_set.begin(),
                                                    value_key_set.end());
  std::sort(values_to_allocate.begin(),
            values_to_allocate.end(),
            [&](DynamicValueAttrs const &lhs, DynamicValueAttrs const &rhs) {
              ValueAllocationInfo const &lhs_info = infos.at(lhs);
              ValueAllocationInfo const &rhs_info = infos.at(rhs);
              if (lhs_info.first_use != rhs_info.first_use) {
                return lhs_info.first_use < rhs_info.first_use;
              }
              return get_allocation_size_bytes(lhs) >
                     get_allocation_size_bytes(rhs);
            });

  std::unordered_map<AllocationPoolKey,
                     std::vector<ReusableAllocation>,
                     AllocationPoolKeyHash>
      reusable_allocations;

  for (DynamicValueAttrs const &value : values_to_allocate) {
    if (contains_key(preallocated, value)) {
      // FIXME: Attach external instance to existing allocation and use that
      NOT_IMPLEMENTED();
    }

    ValueAllocationInfo const &info = infos.at(value);
    ASSERT(info.device_coord.has_value());
    AllocationPoolKey key{
        /*device_coord=*/info.device_coord.value(),
        /*shape=*/get_piece_shape(value.parallel_tensor_shape.value())};

    std::vector<ReusableAllocation> &pool = reusable_allocations[key];
    auto reusable_it =
        std::find_if(pool.begin(), pool.end(), [&](ReusableAllocation &slot) {
          return slot.last_use < info.first_use;
        });

    if (reusable_it == pool.end()) {
      pool.push_back(ReusableAllocation{
          /*instance=*/
          perform_instance_allocation_for_value(key.device_coord, value, ctx),
          /*last_use=*/info.last_use,
      });
      result.backing.insert(std::pair{value, pool.back().instance});
    } else {
      reusable_it->last_use = info.last_use;
      result.backing.insert(std::pair{value, reusable_it->instance});
    }
  }

  print_reused_instance_allocation_summary(result, infos);
  return result;
}

void destroy_instances(TensorInstanceBacking const &instances,
                       Realm::Event precondition) {
  std::unordered_set<Realm::RegionInstance::id_t> destroyed_instances;
  for (auto const &[instance, ready] : values(instances.backing)) {
    if (destroyed_instances.insert(instance.id).second) {
      instance.destroy(Realm::Event::merge_events(precondition, ready));
    }
  }
}

} // namespace FlexFlow
