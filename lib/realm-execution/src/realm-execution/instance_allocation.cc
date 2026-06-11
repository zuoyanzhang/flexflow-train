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
#include <iostream>
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

static MachineSpaceCoordinate
    get_allocation_device_coord(DynamicNodeAttrs const &node_attrs,
                                DynamicValueAttrs const &value) {
  if (value.mapping.has_value() && value.shard_coord.has_value()) {
    return value.mapping.value().at_l(value.shard_coord.value());
  }

  return assert_unwrap(node_attrs.device_coord);
}

static void print_instance_allocation_summary(
    DynamicOpenDataflowGraph const &g,
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

  for (DynamicNodeInvocation const &invocation : g.invocations) {
    for (DynamicValueAttrs const &input : values(invocation.inputs)) {
      visit_value(invocation.node_attrs, input);
    }
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      visit_value(invocation.node_attrs, output);
    }
  }

  std::cerr << "[instance-allocation] invocations=" << g.invocations.size()
            << ", tensor_instances=" << seen_values.size()
            << ", estimated_total_mib=" << bytes_to_mib(total_bytes) << "\n";
  for (auto const &[device_coord, bytes] : bytes_by_device) {
    std::cerr << "[instance-allocation] device=" << device_coord
              << ", estimated_mib=" << bytes_to_mib(bytes) << "\n";
  }
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
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &preallocated,
    RealmContext &ctx) {
  ASSERT(no_tensors_are_allocated(g));
  ASSERT(tensors_are_ready_for_allocation(g));
  for (DynamicValueAttrs const &v : keys(preallocated)) {
    ASSERT(v.accessor == std::nullopt);
  }

  print_instance_allocation_summary(g, preallocated);

  TensorInstanceBacking result = make_empty_tensor_instance_backing();
  auto allocate = [&](DynamicNodeAttrs const &n, DynamicValueAttrs const &v) {
    if (contains_key(preallocated, v)) {
      // FIXME: Attach external instance to existing allocation and use that
      NOT_IMPLEMENTED();
    } else {
      if (!contains_key(result.backing, v)) {
        MachineSpaceCoordinate device_coord = get_allocation_device_coord(n, v);
        result.backing.insert(std::pair{
            v, perform_instance_allocation_for_value(device_coord, v, ctx)});
      }
      return result.backing.at(v);
    }
  };

  for (DynamicNodeInvocation const &invocation : g.invocations) {
    for (DynamicValueAttrs const &input : values(invocation.inputs)) {
      allocate(invocation.node_attrs, input);
    }
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      allocate(invocation.node_attrs, output);
    }
  }

  return result;
}

void destroy_instances(TensorInstanceBacking const &instances,
                       Realm::Event precondition) {
  for (auto const &[instance, ready] : values(instances.backing)) {
    instance.destroy(Realm::Event::merge_events(precondition, ready));
  }
}

} // namespace FlexFlow
