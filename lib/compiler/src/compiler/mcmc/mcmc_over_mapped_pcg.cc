#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/machine_mapping/apply_substitution_and_update_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/mcmc/generic_mcmc_algorithm.h"
#include "compiler/search_result.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/machine_compute_specification.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/optional.h"
#include "utils/overload.h"
#include "utils/random_utils.h"
#include <libassert/assert.hpp>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

namespace {

struct ApplicableSubstitution {
  Substitution substitution;
  PCGPatternMatch match;
};

std::vector<ApplicableSubstitution> get_applicable_substitutions(
    ParallelComputationGraph const &pcg,
    std::vector<Substitution> const &substitutions) {
  std::vector<ApplicableSubstitution> result;
  SubParallelComputationGraph sub_pcg = sub_pcg_from_full_pcg(pcg);

  for (Substitution const &substitution : substitutions) {
    for (PCGPatternMatch const &match :
         find_pattern_matches(substitution.pcg_pattern, sub_pcg)) {
      result.push_back(ApplicableSubstitution{/*substitution=*/substitution,
                                              /*match=*/match});
    }
  }

  return result;
}

bool linear_layer_is_already_parallelized(ParallelComputationGraph const &pcg,
                                          parallel_layer_guid_t layer) {
  for (parallel_layer_guid_t successor : get_successors(pcg, layer)) {
    PCGOperatorAttrs successor_attrs = pcg_get_op_attrs(pcg, successor);
    if (successor_attrs.has<CombineAttrs>() ||
        successor_attrs.has<ReductionAttrs>()) {
      return true;
    }
  }

  return false;
}

bool attention_layer_is_already_parallelized(ParallelComputationGraph const &pcg,
                                             parallel_layer_guid_t layer) {
  for (parallel_layer_guid_t successor : get_successors(pcg, layer)) {
    PCGOperatorAttrs successor_attrs = pcg_get_op_attrs(pcg, successor);
    if (successor_attrs.has<CombineAttrs>() ||
        successor_attrs.has<ReductionAttrs>()) {
      return true;
    }
  }

  return false;
}

bool has_linear_layer(ParallelComputationGraph const &pcg) {
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    if (pcg_get_op_attrs(pcg, layer).has<LinearAttrs>()) {
      return true;
    }
  }

  return false;
}

bool has_attention_layer(ParallelComputationGraph const &pcg) {
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    if (pcg_get_op_attrs(pcg, layer).has<MultiHeadAttentionAttrs>()) {
      return true;
    }
  }

  return false;
}

std::optional<positive_int>
get_single_output_tensor_rank(ParallelComputationGraph const &pcg,
                              parallel_layer_guid_t layer) {
  std::unordered_map<TensorSlotName, parallel_tensor_guid_t> outputs =
      get_outgoing_tensors(pcg, layer);
  auto output_it = outputs.find(TensorSlotName::OUTPUT);
  if (output_it == outputs.end()) {
    return std::nullopt;
  }

  ParallelTensorShape output_shape =
      get_parallel_tensor_shape(pcg, output_it->second);
  int rank = num_shard_dims(output_shape).value.unwrap_nonnegative();
  if (rank <= 0) {
    return std::nullopt;
  }

  return positive_int{rank};
}

std::optional<PatternInput> pattern_input_from_open_value(
    OpenKwargDataflowValue<int, TensorSlotName> const &value) {
  return value.visit<std::optional<PatternInput>>(
      overload{[](KwargDataflowGraphInput<int> const &input) {
                 return std::optional<PatternInput>{PatternInput{input}};
               },
               [](KwargDataflowOutput<TensorSlotName> const &) {
                 return std::optional<PatternInput>{std::nullopt};
               }});
}

std::optional<PCGPatternMatch> make_single_node_pattern_match_for_layer(
    ParallelComputationGraph const &pcg, Substitution const &substitution,
    parallel_layer_guid_t layer, bool validate_assignment = true) {
  std::unordered_set<PatternNode> pattern_nodes =
      get_nodes(substitution.pcg_pattern);
  if (pattern_nodes.size() != 1) {
    return std::nullopt;
  }

  PatternNode pattern_node = *pattern_nodes.begin();
  std::unordered_map<TensorSlotName, parallel_tensor_guid_t> incoming_tensors =
      get_incoming_tensors(pcg, layer);
  std::unordered_map<PatternInput, open_parallel_tensor_guid_t>
      input_assignment;

  for (OpenKwargDataflowEdge<int, TensorSlotName> const &edge :
       get_all_open_kwarg_dataflow_edges(substitution.pcg_pattern.raw_graph)) {
    KwargDataflowInput<TensorSlotName> dst =
        get_dst_of_open_kwarg_dataflow_edge(edge);
    if (dst.node != pattern_node.raw_node) {
      continue;
    }

    std::optional<PatternInput> maybe_pattern_input =
        pattern_input_from_open_value(get_src_of_open_kwarg_dataflow_edge(edge));
    if (!maybe_pattern_input.has_value()) {
      return std::nullopt;
    }

    auto tensor_it = incoming_tensors.find(dst.slot_name);
    if (tensor_it == incoming_tensors.end()) {
      return std::nullopt;
    }
    input_assignment.emplace(
        maybe_pattern_input.value(),
        open_parallel_tensor_guid_from_closed(tensor_it->second));
  }

  bidict<PatternNode, parallel_layer_guid_t> node_assignment;
  node_assignment.equate(pattern_node, layer);
  PCGPatternMatch match{/*node_assignment=*/node_assignment,
                        /*input_assignment=*/input_assignment};

  if (validate_assignment) {
    SubParallelComputationGraph sub_pcg = sub_pcg_from_full_pcg(pcg);
    if (!assignment_satisfies(sub_pcg, substitution.pcg_pattern, match)) {
      return std::nullopt;
    }
  }

  return match;
}

std::vector<ApplicableSubstitution> get_linear_parallelization_candidates(
    ParallelComputationGraph const &pcg,
    MachineComputeSpecification const &compute_spec) {
  std::vector<ApplicableSubstitution> result;
  SubParallelComputationGraph sub_pcg = sub_pcg_from_full_pcg(pcg);
  int max_degree = get_num_gpus(compute_spec).int_from_positive_int();

  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    PCGOperatorAttrs attrs = pcg_get_op_attrs(pcg, layer);
    if (!attrs.has<LinearAttrs>()) {
      continue;
    }
    if (linear_layer_is_already_parallelized(pcg, layer)) {
      continue;
    }

    LinearAttrs linear_attrs = attrs.get<LinearAttrs>();
    std::optional<positive_int> maybe_output_rank =
        get_single_output_tensor_rank(pcg, layer);
    if (!maybe_output_rank.has_value()) {
      continue;
    }

    for (int degree = max_degree; degree > 1; degree /= 2) {
      if (linear_attrs.out_channels.int_from_positive_int() % degree != 0) {
        continue;
      }

      Substitution substitution = create_replicate_linear_combine_unchecked(
          maybe_output_rank.value(), positive_int{degree},
          linear_attrs.use_bias);
      for (PCGPatternMatch const &match :
           find_pattern_matches(substitution.pcg_pattern, sub_pcg)) {
        if (right_entries(match.node_assignment).count(layer) > 0) {
          result.push_back(ApplicableSubstitution{
              /*substitution=*/substitution,
              /*match=*/match,
          });
        }
      }
    }
  }

  return result;
}

std::optional<ApplicableSubstitution> get_linear_parallelization_candidate(
    ParallelComputationGraph const &pcg, parallel_layer_guid_t layer,
    int degree) {
  PCGOperatorAttrs attrs = pcg_get_op_attrs(pcg, layer);
  if (!attrs.has<LinearAttrs>()) {
    return std::nullopt;
  }
  if (linear_layer_is_already_parallelized(pcg, layer)) {
    return std::nullopt;
  }

  LinearAttrs linear_attrs = attrs.get<LinearAttrs>();
  if (linear_attrs.out_channels.int_from_positive_int() % degree != 0) {
    return std::nullopt;
  }

  std::optional<positive_int> maybe_output_rank =
      get_single_output_tensor_rank(pcg, layer);
  if (!maybe_output_rank.has_value()) {
    return std::nullopt;
  }

  Substitution substitution = create_replicate_linear_combine_unchecked(
      maybe_output_rank.value(), positive_int{degree}, linear_attrs.use_bias);
  std::optional<PCGPatternMatch> maybe_match =
      make_single_node_pattern_match_for_layer(pcg, substitution, layer);
  if (maybe_match.has_value()) {
    return ApplicableSubstitution{/*substitution=*/substitution,
                                  /*match=*/maybe_match.value()};
  }

  return std::nullopt;
}

std::optional<ApplicableSubstitution> get_attention_parallelization_candidate(
    ParallelComputationGraph const &pcg, parallel_layer_guid_t layer,
    int degree) {
  PCGOperatorAttrs attrs = pcg_get_op_attrs(pcg, layer);
  if (!attrs.has<MultiHeadAttentionAttrs>()) {
    return std::nullopt;
  }
  if (attention_layer_is_already_parallelized(pcg, layer)) {
    std::cerr << "[mcmc] attention proposal skipped: layer=" << layer
              << ", degree=" << degree << ", reason=already_parallelized\n";
    return std::nullopt;
  }

  MultiHeadAttentionAttrs attention_attrs = attrs.get<MultiHeadAttentionAttrs>();
  if (attention_attrs.embed_dim.int_from_positive_int() % degree != 0 ||
      attention_attrs.num_heads.int_from_positive_int() % degree != 0) {
    std::cerr << "[mcmc] attention proposal skipped: layer=" << layer
              << ", degree=" << degree
              << ", reason=attrs_not_divisible"
              << ", embed_dim=" << attention_attrs.embed_dim
              << ", num_heads=" << attention_attrs.num_heads << "\n";
    return std::nullopt;
  }

  SubParallelComputationGraph sub_pcg = sub_pcg_from_full_pcg(pcg);
  Substitution self_attention_substitution =
      create_replicate_self_attention_reduce(positive_int{degree},
                                             positive_int{degree});
  std::optional<PCGPatternMatch> maybe_self_attention_match =
      make_single_node_pattern_match_for_layer(
          pcg, self_attention_substitution, layer,
          /*validate_assignment=*/false);
  if (maybe_self_attention_match.has_value()) {
    return ApplicableSubstitution{/*substitution=*/self_attention_substitution,
                                  /*match=*/maybe_self_attention_match.value()};
  }

  std::vector<Substitution> attention_substitutions = {
      create_replicate_attention_reduce(positive_int{degree},
                                        positive_int{degree}),
  };
  for (Substitution const &substitution : attention_substitutions) {
    for (PCGPatternMatch const &match :
         find_pattern_matches(substitution.pcg_pattern, sub_pcg)) {
      if (right_entries(match.node_assignment).count(layer) > 0) {
        return ApplicableSubstitution{/*substitution=*/substitution,
                                      /*match=*/match};
      }
    }
  }

  std::cerr << "[mcmc] attention proposal skipped: layer=" << layer
            << ", degree=" << degree << ", reason=no_pattern_match\n";
  return std::nullopt;
}

std::optional<SearchResult> apply_linear_parallelization_block_proposal(
    SearchResult const &mapped_pcg,
    MachineComputeSpecification const &compute_spec, DeviceType device_type,
    int degree) {
  SearchResult result = mapped_pcg;
  std::unordered_set<parallel_layer_guid_t> original_layer_set =
      get_parallel_layers(mapped_pcg.pcg);
  std::vector<parallel_layer_guid_t> original_layers(
      original_layer_set.begin(), original_layer_set.end());
  int transformed = 0;

  for (parallel_layer_guid_t layer : original_layers) {
    if (get_parallel_layers(result.pcg).count(layer) == 0) {
      continue;
    }

    std::optional<ApplicableSubstitution> maybe_candidate =
        get_linear_parallelization_candidate(result.pcg, layer, degree);
    if (!maybe_candidate.has_value()) {
      continue;
    }

    result = apply_substitution_and_update_machine_mapping(
        result, maybe_candidate->substitution, maybe_candidate->match);
    transformed++;
  }

  if (transformed == 0) {
    return std::nullopt;
  }

  std::optional<MachineMapping> maybe_remapped =
      get_random_mapping(result.pcg, compute_spec, device_type);
  if (!maybe_remapped.has_value()) {
    return std::nullopt;
  }

  result = SearchResult{/*pcg=*/result.pcg,
                        /*machine_mapping=*/maybe_remapped.value()};

  return result;
}

std::optional<SearchResult> apply_attention_parallelization_block_proposal(
    SearchResult const &mapped_pcg,
    MachineComputeSpecification const &compute_spec, DeviceType device_type,
    int degree) {
  SearchResult result = mapped_pcg;
  std::unordered_set<parallel_layer_guid_t> original_layer_set =
      get_parallel_layers(mapped_pcg.pcg);
  std::vector<parallel_layer_guid_t> original_layers(
      original_layer_set.begin(), original_layer_set.end());
  int transformed = 0;

  for (parallel_layer_guid_t layer : original_layers) {
    if (get_parallel_layers(result.pcg).count(layer) == 0) {
      continue;
    }

    std::optional<ApplicableSubstitution> maybe_candidate =
        get_attention_parallelization_candidate(result.pcg, layer, degree);
    if (!maybe_candidate.has_value()) {
      continue;
    }

    result = apply_substitution_and_update_machine_mapping(
        result, maybe_candidate->substitution, maybe_candidate->match);
    transformed++;
  }

  if (transformed == 0) {
    return std::nullopt;
  }

  std::optional<MachineMapping> maybe_remapped =
      get_random_mapping(result.pcg, compute_spec, device_type);
  if (!maybe_remapped.has_value()) {
    return std::nullopt;
  }

  result = SearchResult{/*pcg=*/result.pcg,
                        /*machine_mapping=*/maybe_remapped.value()};

  return result;
}

int get_max_combine_parallel_degree_from_pcg(
    ParallelComputationGraph const &pcg) {
  int result = 1;
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    PCGOperatorAttrs attrs = pcg_get_op_attrs(pcg, layer);
    if (attrs.has<CombineAttrs>()) {
      result = std::max(
          result,
          attrs.get<CombineAttrs>().combine_degree.int_from_positive_int());
    }
  }
  return result;
}

} // namespace

SearchResult
    mcmc_over_mapped_pcg(ParallelComputationGraph const &pcg,
                         RuntimeOnlyCostEstimator const &cost_estimator,
                         MachineSpecification const &machine_spec,
                         MCMCOverMappedPCGConfig const &search_config) {
  MachineComputeSpecification compute_spec = machine_spec.compute_specification;
  std::vector<Substitution> substitutions = get_substitution_set(compute_spec);
  MachineMapping random_mapping = assert_unwrap(
      get_random_mapping(pcg, compute_spec, search_config.device_type));
  SearchResult starting_state = SearchResult{pcg, random_mapping};

  auto cost = [&](SearchResult mapped_pcg) -> float {
    return task_simulator_estimate_forward_pass_time(mapped_pcg.pcg,
                                                     cost_estimator,
                                                     mapped_pcg.machine_mapping,
                                                     machine_spec)
        .unwrap_milliseconds();
  };

  auto sampler = [&](SearchResult mapped_pcg) -> std::optional<SearchResult> {
    // applies substitution with substitution_frequency probability
    // applies machine mapping mutation with (1 - substitution_frequency)
    // probability
    ASSERT(search_config.substitution_frequency >= 0 &&
           search_config.substitution_frequency <= 1);
    if (randf() < search_config.substitution_frequency) {
      std::optional<SearchResult> best_soap_block = std::nullopt;
      float best_soap_block_cost = std::numeric_limits<float>::infinity();
      std::string best_soap_block_kind;
      int best_soap_block_degree = 1;
      int max_degree = get_num_gpus(compute_spec).int_from_positive_int();
      auto consider_candidate =
          [&](char const *kind, int degree,
              std::optional<SearchResult> const &maybe_candidate) {
            if (!maybe_candidate.has_value()) {
              std::cerr << "[mcmc] soap proposal unavailable: kind=" << kind
                        << ", degree=" << degree << "\n";
              return;
            }

            float candidate_cost = cost(maybe_candidate.value());
            std::cerr << "[mcmc] soap proposal candidate: kind=" << kind
                      << ", degree=" << degree
                      << ", cost_ms=" << candidate_cost << "\n";
            if (candidate_cost < best_soap_block_cost) {
              best_soap_block = maybe_candidate;
              best_soap_block_cost = candidate_cost;
              best_soap_block_kind = kind;
              best_soap_block_degree = degree;
            }
          };

      for (int degree = max_degree; degree > 1; degree /= 2) {
        std::optional<SearchResult> maybe_candidate =
            apply_linear_parallelization_block_proposal(
                mapped_pcg, compute_spec, search_config.device_type, degree);
        consider_candidate("linear", degree, maybe_candidate);

        maybe_candidate = apply_attention_parallelization_block_proposal(
            mapped_pcg, compute_spec, search_config.device_type, degree);
        consider_candidate("attention", degree, maybe_candidate);
      }

      if (best_soap_block.has_value()) {
        std::cerr << "[mcmc] soap proposal selected: kind="
                  << best_soap_block_kind
                  << ", degree=" << best_soap_block_degree
                  << ", cost_ms=" << best_soap_block_cost << "\n";
        return best_soap_block;
      }

      std::vector<ApplicableSubstitution> applicable_substitutions;
      if (!has_linear_layer(mapped_pcg.pcg) &&
          !has_attention_layer(mapped_pcg.pcg)) {
        applicable_substitutions =
            get_applicable_substitutions(mapped_pcg.pcg, substitutions);
      }
      if (applicable_substitutions.empty()) {
        return std::nullopt;
      }

      ApplicableSubstitution applicable_substitution =
          select_random(applicable_substitutions);
      SearchResult substituted = apply_substitution_and_update_machine_mapping(
          mapped_pcg, applicable_substitution.substitution,
          applicable_substitution.match);
      std::optional<MachineMapping> maybe_remapped =
          get_random_mapping(substituted.pcg, compute_spec,
                             search_config.device_type);
      if (!maybe_remapped.has_value()) {
        return std::nullopt;
      }

      substituted = SearchResult{substituted.pcg, maybe_remapped.value()};

      return substituted;
    } else {
      std::optional<MachineMapping> maybe_new_machine_mapping =
          get_random_mutation(
              mapped_pcg, compute_spec, search_config.device_type);
      return transform(
          maybe_new_machine_mapping,
          [&](MachineMapping const &new_machine_mapping) -> SearchResult {
            return SearchResult{mapped_pcg.pcg, new_machine_mapping};
          });
    }
  };

  GenericMCMCConfig config =
      GenericMCMCConfig{/*temperature*/ search_config.temperature,
                        /*num_iterations*/ search_config.num_iterations};

  SearchResult result = run_mcmc(starting_state, sampler, cost, config);

  return result;
}

} // namespace FlexFlow
