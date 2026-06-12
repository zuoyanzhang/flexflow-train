#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "compiler/machine_mapping/machine_view.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/optional.h"
#include "utils/random_utils.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace FlexFlow {

static std::vector<MachineView> keep_machine_views_with_maximal_device_coverage(
    OperatorTaskSpace const &task_space,
    std::vector<MachineView> const &machine_views) {
  size_t max_num_devices = 0;
  for (MachineView const &machine_view : machine_views) {
    max_num_devices = std::max(
        max_num_devices,
        get_machine_space_coordinates(task_space, machine_view).size());
  }

  std::vector<MachineView> result;
  for (MachineView const &machine_view : machine_views) {
    if (get_machine_space_coordinates(task_space, machine_view).size() ==
        max_num_devices) {
      result.push_back(machine_view);
    }
  }
  return result;
}

static MachineView select_least_loaded_machine_view(
    OperatorTaskSpace const &task_space,
    std::vector<MachineView> const &machine_views,
    std::unordered_map<MachineSpaceCoordinate, size_t> const
        &layers_by_device) {
  auto get_device_load = [&](MachineSpaceCoordinate const &device_coord) {
    auto it = layers_by_device.find(device_coord);
    if (it == layers_by_device.end()) {
      return size_t{0};
    }
    return it->second;
  };

  auto get_view_score = [&](MachineView const &machine_view) {
    std::unordered_set<MachineSpaceCoordinate> device_coords =
        get_machine_space_coordinates(task_space, machine_view);

    size_t max_device_load = 0;
    size_t total_device_load = 0;
    for (MachineSpaceCoordinate const &device_coord : device_coords) {
      size_t device_load = get_device_load(device_coord);
      max_device_load = std::max(max_device_load, device_load);
      total_device_load += device_load;
    }

    return std::pair{max_device_load, total_device_load};
  };

  return *std::min_element(
      machine_views.begin(),
      machine_views.end(),
      [&](MachineView const &lhs, MachineView const &rhs) {
        return get_view_score(lhs) < get_view_score(rhs);
      });
}

std::optional<MachineMapping>
get_random_mapping(ParallelComputationGraph const &pcg,
                   MachineComputeSpecification const &resources,
                   DeviceType const &device_type) {
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  std::unordered_map<parallel_layer_guid_t, MachineView> machine_views;
  std::unordered_map<MachineSpaceCoordinate, size_t> layers_by_device;
  for (parallel_layer_guid_t layer : layers) {
    OperatorTaskSpace task = get_operator_task_space(pcg, layer);
    std::unordered_set<MachineView> allowed_machine_views =
        get_allowed_machine_views(compute_slice_from_specification(resources),
                                  task, DeviceType::GPU);
    if (allowed_machine_views.empty()) {
      return std::nullopt;
    }

    ComputationGraphOpAttrs op_attrs = assert_unwrap(
        compgraph_op_attrs_from_pcg_op_attrs(pcg_get_op_attrs(pcg, layer)));
    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
        inputs_dim_degrees = get_incoming_input_degrees(pcg, layer);

    std::vector<MachineView> materializable_machine_views;
    std::optional<std::string> first_materialization_error;
    for (MachineView const &machine_view : allowed_machine_views) {
      try {
        (void)mapped_operator_task_group_from_machine_view(
            op_attrs, inputs_dim_degrees, machine_view);
        materializable_machine_views.push_back(machine_view);
      } catch (std::exception const &e) {
        if (!first_materialization_error.has_value()) {
          first_materialization_error = e.what();
        }
      }
    }

    if (materializable_machine_views.empty()) {
      std::optional<std::string> layer_name =
          get_parallel_layer_attrs(pcg, layer).name;
      std::cerr << "[machine-mapping] no materializable machine view for layer "
                << layer_name.value_or("<unnamed>")
                << ", allowed_views=" << allowed_machine_views.size();
      if (first_materialization_error.has_value()) {
        std::cerr << ", first_error=" << first_materialization_error.value();
      }
      std::cerr << "\n";
      return std::nullopt;
    }

    materializable_machine_views =
        keep_machine_views_with_maximal_device_coverage(
            task, materializable_machine_views);

    MachineView selected_machine_view = select_least_loaded_machine_view(
        task, materializable_machine_views, layers_by_device);
    machine_views.insert({layer, selected_machine_view});

    for (MachineSpaceCoordinate const &device_coord :
         get_machine_space_coordinates(task, selected_machine_view)) {
      layers_by_device[device_coord]++;
    }
  }
  return MachineMapping{machine_views};
}

std::optional<MachineMapping>
get_random_mutation(SearchResult const &mapped_pcg,
                    MachineComputeSpecification const &resources,
                    DeviceType const &device_type) {
  ParallelComputationGraph pcg = mapped_pcg.pcg;
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  if (layers.size() == 0) {
    return std::nullopt;
  }
  parallel_layer_guid_t random_layer = select_random(layers);

  MachineMapping machine_mapping = mapped_pcg.machine_mapping;
  MachineView machine_view = machine_mapping.machine_views.at(random_layer);
  OperatorTaskSpace task = get_operator_task_space(pcg, random_layer);

  std::vector<MachineView> allowed_machine_views =
      vector_of(get_allowed_machine_views(
          compute_slice_from_specification(resources), task, device_type));
  MachineView random_new_machine_view = select_random(allowed_machine_views);

  machine_mapping.machine_views.at(random_layer) = random_new_machine_view;
  return machine_mapping;
}
} // namespace FlexFlow
