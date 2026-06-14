#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/task_space_coordinate.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "utils/bidict/algorithms/bidict_from_map.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/keys.h"

namespace FlexFlow {

static nonnegative_int clamp_parallel_coord_component(nonnegative_int coord,
                                                      positive_int degree) {
  int raw_coord = coord.unwrap_nonnegative();
  int raw_degree = degree.int_from_positive_int();
  return nonnegative_int{raw_degree <= 1 ? 0 : raw_coord % raw_degree};
}

static ParallelTensorSpaceCoordinate project_parallel_coord_to_degrees(
    ParallelTensorSpaceCoordinate const &coord,
    ParallelTensorDimDegrees const &degrees) {
  std::vector<nonnegative_int> shard_components;
  shard_components.reserve(degrees.shard_degrees.size());
  for (size_t idx = 0; idx < degrees.shard_degrees.size(); idx++) {
    nonnegative_int input_coord =
        idx < coord.shard_components.size()
            ? coord.shard_components.at(ff_dim_t{nonnegative_int{idx}})
            : nonnegative_int{0};
    shard_components.push_back(
        clamp_parallel_coord_component(input_coord,
                                       degrees.shard_degrees.at(
                                           ff_dim_t{nonnegative_int{idx}})));
  }

  return ParallelTensorSpaceCoordinate{
      clamp_parallel_coord_component(coord.sum_component,
                                     degrees.sum_degree.value),
      clamp_parallel_coord_component(coord.discard_copy_component,
                                     degrees.discard_copy_degree.value),
      FFOrdered<nonnegative_int>(shard_components.cbegin(),
                                 shard_components.cend()),
  };
}

static std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
get_all_slot_parallel_degrees(ParallelComputationGraph const &pcg,
                              parallel_layer_guid_t layer) {
  std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> result;
  for (auto const &[slot_name, tensor] : get_incoming_tensors(pcg, layer)) {
    result.insert({slot_name,
                   get_parallel_degrees(get_parallel_tensor_shape(pcg,
                                                                  tensor))});
  }
  for (auto const &[slot_name, tensor] : get_outgoing_tensors(pcg, layer)) {
    result.insert({slot_name,
                   get_parallel_degrees(get_parallel_tensor_shape(pcg,
                                                                  tensor))});
  }
  return result;
}

static MappedOperatorTaskGroup mapped_parallel_op_task_group_from_machine_view(
    ParallelComputationGraph const &pcg, parallel_layer_guid_t layer,
    MachineView const &machine_view) {
  PCGOperatorAttrs pcg_op_attrs = pcg_get_op_attrs(pcg, layer);
  OperatorTaskSpace task = get_operator_task_space(pcg, layer);
  parallel_tensor_guid_t output_tensor =
      get_outgoing_tensors(pcg, layer).at(TensorSlotName::OUTPUT);
  ParallelTensorDimDegrees output_degrees =
      get_parallel_degrees(get_parallel_tensor_shape(pcg, output_tensor));
  OperatorSpaceToParallelTensorSpaceMapping output_mapping =
      get_identity_mapping(task, output_degrees);
  num_ptensor_shard_dims_t num_shard_dims =
      get_ptensor_dim_degrees_num_shard_dims(output_degrees);
  std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> slot_degrees =
      get_all_slot_parallel_degrees(pcg, layer);
  bool project_input_slots = pcg_op_attrs.has<CombineAttrs>() ||
                             pcg_op_attrs.has<ReductionAttrs>();

  return MappedOperatorTaskGroup{
      generate_bidict(
          get_machine_space_coordinates(task, machine_view),
          [&](MachineSpaceCoordinate const &machine_space_coord) {
            TaskSpaceCoordinate task_space_coord =
                mv_task_space_coord_for_machine_space_coord(
                    machine_view, task, machine_space_coord);
            ParallelTensorSpaceCoordinate tensor_coord =
                ptensor_coord_for_task_space_coord(
                    output_mapping, task_space_coord, num_shard_dims);

            std::unordered_map<TensorSlotName, ParallelTensorSpaceCoordinate>
                tensor_coords;
            for (auto const &[slot_name, degrees] : slot_degrees) {
              bool is_output = slot_name == TensorSlotName::OUTPUT;
              tensor_coords.insert({
                  slot_name,
                  (project_input_slots || is_output)
                      ? project_parallel_coord_to_degrees(tensor_coord, degrees)
                      : tensor_coord,
              });
            }

            return OperatorAtomicTaskShardBinding{
                /*tensor_coords=*/tensor_coords,
            };
          }),
  };
}

MappedParallelComputationGraph
mapped_pcg_from_pcg_and_mapping(ParallelComputationGraph const &pcg,
                                MachineMapping const &mapping) {

  std::unordered_set<parallel_layer_guid_t> pcg_layers =
      get_parallel_layers(pcg);

  std::unordered_set<parallel_layer_guid_t> mapped_layers =
      keys(mapping.machine_views);

  ASSERT(mapped_layers == pcg_layers);

  auto mapping_for_layer =
      [&](parallel_layer_guid_t l) -> MappedOperatorTaskGroup {
    PCGOperatorAttrs pcg_op_attrs = pcg_get_op_attrs(pcg, l);

    ASSERT(contains_key(mapping.machine_views, l));
    MachineView machine_view = mapping.machine_views.at(l);

    if (is_parallel_op(pcg_op_attrs)) {
      return mapped_parallel_op_task_group_from_machine_view(pcg, l,
                                                             machine_view);
    }

    ComputationGraphOpAttrs op_attrs =
        assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(pcg_op_attrs));

    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
        inputs_dim_degrees = get_incoming_input_degrees(pcg, l);

    return mapped_operator_task_group_from_machine_view(
        op_attrs, inputs_dim_degrees, machine_view);
  };

  std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup>
      mapped_op_task_groups = generate_map(mapped_layers, mapping_for_layer);

  return mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg,
                                                       mapped_op_task_groups);
}

MachineMapping combine_disjoint_mappings(MachineMapping const &m1,
                                         MachineMapping const &m2) {
  return MachineMapping{
      binary_merge_disjoint_maps(m1.machine_views, m2.machine_views),
  };
}

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2) {
  return are_disjoint(keys(m1.machine_views), keys(m2.machine_views));
}

std::optional<MachineMapping> get_machine_mapping_from_machine_mapping_result(
    PCGBinarySPDecomposition const &sp_decomposition,
    MachineMappingResult const &mm_result) {

  FeasibleMachineMappingResult feasible_mapping = ({
    if (is_infeasible(mm_result)) {
      return std::nullopt;
    }

    require_feasible(mm_result);
  });

  bidict<BinaryTreePath, parallel_layer_guid_t> path_to_leaf_map =
      bidict_from_map(pcg_sp_tree_get_path_to_leaf_map(sp_decomposition));

  return MachineMapping{
      map_keys(feasible_mapping.machine_mapping.raw_mapping,
               [&](BinaryTreePath const &p) -> parallel_layer_guid_t {
                 return path_to_leaf_map.at_l(p);
               }),
  };
}

} // namespace FlexFlow
