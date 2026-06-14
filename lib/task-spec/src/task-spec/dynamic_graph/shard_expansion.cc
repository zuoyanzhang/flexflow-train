#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/filter_keys.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/require_same.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include <algorithm>
#include <iostream>
#include <optional>
#include <sstream>

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &n) {
  return n.device_coord.has_value();
}

bool value_is_shard_expanded(DynamicValueAttrs const &n) {
  return n.shard_coord.has_value();
}

bool no_part_of_graph_is_shard_expanded(DynamicOpenDataflowGraph const &g) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) -> bool {
    return false;
  };

  return no_part_of_dynamic_graph_satisfies(g, node_is_shard_expanded,
                                            value_is_shard_expanded,
                                            slot_is_shard_expanded);
}

bool graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &g) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) -> bool {
    return true;
  };

  return full_dynamic_graph_satisfies(g, node_is_shard_expanded,
                                      value_is_shard_expanded,
                                      slot_is_shard_expanded);
}

static bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
restrict_tensor_mapping_keys_to_coord(
    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const
        &mapping,
    ParallelTensorSpaceCoordinate const &parallel_tensor_coord) {
  return filter_keys(mapping, [&](ParallelTensorSpaceCoordinate const &p) {
    return p == parallel_tensor_coord;
  });
}

static DynamicNodeInvocation
shard_invocation_for_binding(DynamicNodeInvocation const &i,
                             MachineSpaceCoordinate const &machine_coord,
                             OperatorAtomicTaskShardBinding const &binding) {
  auto shard_expand_value_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrs {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    DynamicValueAttrs result = v;
    result.shard_coord = parallel_tensor_coord;
    result.mapping = transform(
        v.mapping, [&](bidict<ParallelTensorSpaceCoordinate,
                              MachineSpaceCoordinate> const &mapping) {
          return restrict_tensor_mapping_keys_to_coord(mapping,
                                                       parallel_tensor_coord);
        });
    return result;
  };

  DynamicNodeAttrs expanded_node_attrs = [&]() {
    DynamicNodeAttrs result = i.node_attrs;
    result.device_coord = machine_coord;
    return result;
  }();

  return DynamicNodeInvocation{
      /*inputs=*/map_values2(i.inputs, shard_expand_value_attrs),
      /*node_attrs=*/expanded_node_attrs,
      /*outputs=*/map_values2(i.outputs, shard_expand_value_attrs),
  };
}

static std::unordered_set<DynamicNodeInvocation>
perform_shard_expansion_for_copy(DynamicNodeInvocation const &i) {
  auto [input_slot, input] = get_only(i.inputs);
  auto [output_slot, output] = get_only(i.outputs);
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      assert_unwrap(input.mapping);
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> output_mapping =
      assert_unwrap(output.mapping);

  auto make_copy_shard =
      [&](ParallelTensorSpaceCoordinate const &input_coord,
          ParallelTensorSpaceCoordinate const &output_coord) {
        // The machine coord for a copy is inherently nebulous because it
        // doesn't strictly run in any single location. Further, Realm has the
        // flexibility to issue a copy operation from anywhere in the machine,
        // including remotely. Here we choose machine_coord based on the input
        // because we expect this to align with the most efficient way to issue
        // copies in Realm, although the current Realm backend uses a
        // centralized controller and thus issues copies all from a single node.
        MachineSpaceCoordinate machine_coord = input_mapping.at_l(input_coord);

        return shard_invocation_for_binding(
            i, machine_coord,
            OperatorAtomicTaskShardBinding{{
                {input_slot.slot_name, input_coord},
                {output_slot.slot_name, output_coord},
            }});
      };

  if (input_mapping.left_values() == output_mapping.left_values()) {
    return transform(input_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &p) {
                       return make_copy_shard(p, p);
                     });
  }

  if (input_mapping.left_values().size() == 1) {
    ParallelTensorSpaceCoordinate input_coord =
        get_only(input_mapping.left_values());
    return transform(output_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &output_coord) {
                       return make_copy_shard(input_coord, output_coord);
                     });
  }

  if (output_mapping.left_values().size() == 1) {
    ParallelTensorSpaceCoordinate output_coord =
        get_only(output_mapping.left_values());
    return transform(input_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &input_coord) {
                       return make_copy_shard(input_coord, output_coord);
                     });
  }

  if (input_mapping.left_values().empty() ||
      output_mapping.left_values().empty()) {
    return {};
  }

  bool output_coords_are_input_subset = true;
  for (ParallelTensorSpaceCoordinate const &output_coord :
       output_mapping.left_values()) {
    if (input_mapping.left_values().count(output_coord) == 0) {
      output_coords_are_input_subset = false;
      break;
    }
  }
  if (output_coords_are_input_subset) {
    return transform(output_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &coord) {
                       return make_copy_shard(coord, coord);
                     });
  }

  bool input_coords_are_output_subset = true;
  for (ParallelTensorSpaceCoordinate const &input_coord :
       input_mapping.left_values()) {
    if (output_mapping.left_values().count(input_coord) == 0) {
      input_coords_are_output_subset = false;
      break;
    }
  }
  if (input_coords_are_output_subset) {
    auto representative_input_coord =
        [&](ParallelTensorSpaceCoordinate const &output_coord) {
          if (input_mapping.left_values().count(output_coord) > 0) {
            return output_coord;
          }

          std::optional<ParallelTensorSpaceCoordinate> best = std::nullopt;
          int best_score = -1;
          for (ParallelTensorSpaceCoordinate const &input_coord :
               input_mapping.left_values()) {
            int score = 0;
            if (input_coord.sum_component == output_coord.sum_component) {
              score++;
            }
            if (input_coord.discard_copy_component ==
                output_coord.discard_copy_component) {
              score++;
            }
            size_t common_shard_dims =
                std::min(input_coord.shard_components.size(),
                         output_coord.shard_components.size());
            for (size_t idx = 0; idx < common_shard_dims; idx++) {
              if (input_coord.shard_components.at(
                      ff_dim_t{nonnegative_int{idx}}) ==
                  output_coord.shard_components.at(
                      ff_dim_t{nonnegative_int{idx}})) {
                score++;
              }
            }
            if (!best.has_value() || score > best_score) {
              best = input_coord;
              best_score = score;
            }
          }
          return assert_unwrap(best);
        };

    return transform(output_mapping.left_values(),
                     [&](ParallelTensorSpaceCoordinate const &output_coord) {
                       return make_copy_shard(
                           representative_input_coord(output_coord),
                           output_coord);
                     });
  }

  auto format_coord_set =
      [](std::unordered_set<ParallelTensorSpaceCoordinate> const &coords) {
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (ParallelTensorSpaceCoordinate const &coord : coords) {
          oss << (first ? "" : ",") << coord;
          first = false;
        }
        oss << "}";
        return oss.str();
      };
  std::cerr << "[shard-expansion] unsupported multi-source/multi-dest copy"
            << ": input_coords=" << format_coord_set(input_mapping.left_values())
            << ", output_coords="
            << format_coord_set(output_mapping.left_values()) << "\n";
  require_same(input_mapping.left_values(), output_mapping.left_values());
  return {};
}

std::unordered_set<DynamicNodeInvocation>
perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {
  if (i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().is_copy()) {
    return perform_shard_expansion_for_copy(i);
  }

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  std::unordered_set<MachineSpaceCoordinate> shard_machine_coords =
      mapping.get_shard_bindings().left_values();

  return transform(
      shard_machine_coords,
      [&](MachineSpaceCoordinate const &c) -> DynamicNodeInvocation {
        OperatorAtomicTaskShardBinding slot_bindings =
            mapping.get_shard_bindings().at_l(c);

        return shard_invocation_for_binding(i, c, slot_bindings);
      });
}

DynamicOpenDataflowGraph
perform_shard_expansion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_shard_expanded(g));

  std::cerr << "[shard-expansion] start: invocations="
            << g.invocations.size() << "\n";
  std::unordered_set<DynamicNodeInvocation> expanded_invocations;
  expanded_invocations.reserve(g.invocations.size() * 4);
  size_t processed_invocations = 0;
  for (DynamicNodeInvocation const &i : g.invocations) {
    if (processed_invocations % 500 == 0) {
      std::cerr << "[shard-expansion] progress: invocation "
                << processed_invocations << "/" << g.invocations.size()
                << ", expanded=" << expanded_invocations.size() << "\n";
    }
    std::unordered_set<DynamicNodeInvocation> expanded =
        perform_shard_expansion_for_invocation(i);
    expanded_invocations.insert(expanded.cbegin(), expanded.cend());
    processed_invocations++;
  }
  std::cerr << "[shard-expansion] expansion done: expanded="
            << expanded_invocations.size() << "\n";

  DynamicOpenDataflowGraph result =
      dynamic_open_dataflow_graph_from_invocation_set(expanded_invocations);

  std::cerr << "[shard-expansion] validation start\n";
  ASSERT(graph_is_fully_shard_expanded(result));
  std::cerr << "[shard-expansion] validation done\n";

  return result;
}

} // namespace FlexFlow
