#include "compiler/cost_estimator/runtime_only_cost_estimator.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/mcmc/mcmc_over_mapped_pcg.h"
#include "compiler/unity_algorithm/unity_algorithm.h"
#include "models/transformer/transformer.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_dims.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/machine_compute_specification.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/machine_interconnect_specification.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/pcg_instance.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_manager.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/bidict.h"
#include "utils/containers/keys.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/overload.h"
#include "utils/orthotope/down_projection.h"
#include "utils/positive_int/positive_int.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace FlexFlow;

namespace {

enum class MappingMode {
  BALANCED,
  UNITY,
  MCMC,
};

struct BenchmarkArgs {
  int num_cpus = 4;
  int num_nodes = 1;
  int num_gpus = 1;
  int fbmem_mb = 64000;
  int zcmem_mb = 4096;
  int workspace_mb = 1024;
  int batch_size = 1;
  int sequence_length = 2048;
  int decoder_layers = 32;
  int warmup_iters = 1;
  int measure_iters = 5;
  MappingMode mapping_mode = MappingMode::BALANCED;
  bool mapping_mode_explicit = false;
  int search_budget = 0;
  int search_decoder_layers = 0;
  float search_alpha = 0.05f;
  int search_max_num_ops = 100000;
  float mcmc_temperature = 1.0f;
  float mcmc_substitution_frequency = 0.2f;
  float inter_node_bandwidth_gbps = 12.5f;
  float intra_node_bandwidth_gbps = 100.0f;
  int data_parallel_degree = 1;
  int linear_parallel_degree = 1;
  int attention_parallel_degree = 0;
};

char *leak_string_contents(std::string_view str) {
  std::vector<char> *content = new std::vector<char>{str.begin(), str.end()};
  content->push_back(0);
  return content->data();
}

std::vector<char *> make_realm_args(std::string_view executable_name,
                                    BenchmarkArgs const &args) {
  std::vector<std::string> values = {
      std::string{executable_name},  "-ll:cpu",
      std::to_string(args.num_cpus), "-ll:gpu",
      std::to_string(args.num_gpus), "-ll:fsize",
      std::to_string(args.fbmem_mb), "-ll:zsize",
      std::to_string(args.zcmem_mb),
  };

  std::vector<char *> result;
  for (std::string const &value : values) {
    result.push_back(leak_string_contents(value));
  }
  return result;
}

void print_usage(std::string_view prog_name) {
  std::cerr << "usage: " << prog_name
            << " [--nodes N] [--gpus N] [--cpus N] [--fbmem-mb MB]"
               " [--zcmem-mb MB]"
               " [--workspace-mb MB] [--batch-size N]"
               " [--seq-len N] [--decoder-layers N]"
               " [--warmup N] [--iters N]"
               " [--mapping balanced|unity|mcmc]"
               " [--search-budget N] [--search-alpha F]"
               " [--search-decoder-layers N]"
               " [--search-max-num-ops N]"
               " [--mcmc-temperature F]"
               " [--mcmc-substitution-frequency F]"
               " [--inter-node-bandwidth-gbps F]"
               " [--intra-node-bandwidth-gbps F]"
               " [--data-parallel-degree N]"
               " [--linear-parallel-degree N]"
               " [--attention-parallel-degree N]\n";
}

int total_num_gpus(BenchmarkArgs const &args) {
  return args.num_nodes * args.num_gpus;
}

int parse_positive_int_arg(std::string const &name, std::string const &value) {
  int parsed = std::stoi(value);
  if (parsed <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return parsed;
}

int parse_nonnegative_int_arg(std::string const &name,
                              std::string const &value) {
  int parsed = std::stoi(value);
  if (parsed < 0) {
    throw std::invalid_argument(name + " must be nonnegative");
  }
  return parsed;
}

float parse_positive_float_arg(std::string const &name,
                               std::string const &value) {
  float parsed = std::stof(value);
  if (parsed <= 0.0f) {
    throw std::invalid_argument(name + " must be positive");
  }
  return parsed;
}

float parse_probability_arg(std::string const &name, std::string const &value) {
  float parsed = std::stof(value);
  if (parsed < 0.0f || parsed > 1.0f) {
    throw std::invalid_argument(name + " must be in [0, 1]");
  }
  return parsed;
}

MappingMode parse_mapping_mode(std::string const &value) {
  if (value == "balanced") {
    return MappingMode::BALANCED;
  }
  if (value == "unity") {
    return MappingMode::UNITY;
  }
  if (value == "mcmc") {
    return MappingMode::MCMC;
  }
  throw std::invalid_argument(
      "--mapping must be one of: balanced, unity, mcmc");
}

std::string mapping_mode_name(MappingMode mode) {
  switch (mode) {
  case MappingMode::BALANCED:
    return "balanced";
  case MappingMode::UNITY:
    return "unity";
  case MappingMode::MCMC:
    return "mcmc";
  }
  throw std::invalid_argument("unknown mapping mode");
}

BenchmarkArgs parse_args(int argc, char **argv) {
  BenchmarkArgs result;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto read_value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(arg + " requires a value");
      }
      i++;
      return argv[i];
    };

    if (arg == "--nodes" || arg == "--num-nodes") {
      result.num_nodes = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--gpus") {
      result.num_gpus = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--cpus") {
      result.num_cpus = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--fbmem-mb") {
      result.fbmem_mb = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--zcmem-mb") {
      result.zcmem_mb = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--workspace-mb") {
      result.workspace_mb = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--batch-size") {
      result.batch_size = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--seq-len") {
      result.sequence_length = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--decoder-layers") {
      result.decoder_layers = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--warmup") {
      result.warmup_iters = parse_nonnegative_int_arg(arg, read_value());
    } else if (arg == "--iters") {
      result.measure_iters = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--mapping") {
      result.mapping_mode = parse_mapping_mode(read_value());
      result.mapping_mode_explicit = true;
    } else if (arg == "--search-budget" || arg == "--budget") {
      result.search_budget = parse_nonnegative_int_arg(arg, read_value());
    } else if (arg == "--search-decoder-layers") {
      result.search_decoder_layers =
          parse_nonnegative_int_arg(arg, read_value());
    } else if (arg == "--search-alpha" || arg == "--alpha") {
      result.search_alpha = parse_positive_float_arg(arg, read_value());
    } else if (arg == "--search-max-num-ops") {
      result.search_max_num_ops = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--mcmc-temperature") {
      result.mcmc_temperature = parse_positive_float_arg(arg, read_value());
    } else if (arg == "--mcmc-substitution-frequency") {
      result.mcmc_substitution_frequency =
          parse_probability_arg(arg, read_value());
    } else if (arg == "--inter-node-bandwidth-gbps") {
      result.inter_node_bandwidth_gbps =
          parse_positive_float_arg(arg, read_value());
    } else if (arg == "--intra-node-bandwidth-gbps") {
      result.intra_node_bandwidth_gbps =
          parse_positive_float_arg(arg, read_value());
    } else if (arg == "--data-parallel-degree" ||
               arg == "--sample-parallel-degree") {
      result.data_parallel_degree = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--linear-parallel-degree") {
      result.linear_parallel_degree = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--attention-parallel-degree") {
      result.attention_parallel_degree =
          parse_positive_int_arg(arg, read_value());
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (!result.mapping_mode_explicit && result.search_budget > 0) {
    result.mapping_mode = MappingMode::MCMC;
  }
  if (result.search_decoder_layers > result.decoder_layers) {
    throw std::invalid_argument(
        "--search-decoder-layers cannot exceed --decoder-layers");
  }
  if (result.search_decoder_layers > 0 &&
      result.mapping_mode != MappingMode::MCMC) {
    throw std::invalid_argument(
        "--search-decoder-layers currently requires --mapping mcmc");
  }
  if (result.linear_parallel_degree > total_num_gpus(result)) {
    throw std::invalid_argument(
        "--linear-parallel-degree cannot exceed --nodes * --gpus");
  }
  if (result.attention_parallel_degree > total_num_gpus(result)) {
    throw std::invalid_argument(
        "--attention-parallel-degree cannot exceed --nodes * --gpus");
  }
  if (result.data_parallel_degree > total_num_gpus(result)) {
    throw std::invalid_argument(
        "--data-parallel-degree cannot exceed --nodes * --gpus");
  }
  if (result.batch_size % result.data_parallel_degree != 0) {
    throw std::invalid_argument(
        "--batch-size must be divisible by --data-parallel-degree");
  }
  int local_batch_size = result.batch_size / result.data_parallel_degree;
  if (result.mapping_mode != MappingMode::MCMC &&
      result.data_parallel_degree > 1 && local_batch_size > 1 &&
      result.sequence_length >= 1024 && result.decoder_layers >= 16) {
    throw std::invalid_argument(
        "this LLaMA benchmark uses cuDNN MHA, whose training reserve memory is "
        "too large for local batch > 1 at seq_len >= 1024 and >= 16 decoder "
        "layers. Use --batch-size <= --data-parallel-degree, e.g. "
        "--batch-size 4 --data-parallel-degree 4 on a 4x4 job.");
  }
  if (result.mapping_mode != MappingMode::MCMC &&
      result.data_parallel_degree == total_num_gpus(result) &&
      result.decoder_layers >= 16) {
    throw std::invalid_argument(
        "--data-parallel-degree equal to all GPUs creates pure data "
        "parallelism and replicates the full 7B-like model on every GPU; this "
        "does not fit the current FP32 training benchmark for >= 16 decoder "
        "layers. Use a smaller data-parallel degree with rotated layer "
        "placement, e.g. --batch-size 4 --data-parallel-degree 4.");
  }
  return result;
}

MachineInterconnectSpecification
make_machine_interconnect_specification(BenchmarkArgs const &args) {
  auto gbps_to_bytes_per_second = [](float gbps) {
    return bytes_per_second_t{gbps * 1000.0f * 1000.0f * 1000.0f};
  };

  return MachineInterconnectSpecification{
      /*inter_node_bandwidth=*/
      gbps_to_bytes_per_second(args.inter_node_bandwidth_gbps),
      /*intra_node_bandwidth=*/
      gbps_to_bytes_per_second(args.intra_node_bandwidth_gbps),
  };
}

size_t unwrap_num_bytes(num_bytes_t bytes) {
  return bytes.unwrap_num_bytes().size_t_from_nonnegative_int();
}

size_t get_total_piece_size_in_bytes(
    std::unordered_map<TensorSlotName, ParallelTensorShape> const &shapes) {
  size_t result = 0;
  for (auto const &[_, shape] : shapes) {
    result += unwrap_num_bytes(get_piece_size_in_bytes(shape));
  }
  return result;
}

struct HeuristicRuntimeOnlyCostEstimator : public IRuntimeOnlyCostEstimator {
  explicit HeuristicRuntimeOnlyCostEstimator(
      MachineInterconnectSpecification const &interconnect_specification)
      : interconnect_specification(interconnect_specification) {}

  RuntimeOnlyOpCostMetrics
  estimate_cost(RuntimeOnlyOpCostEstimateKey const &key) const override {
    size_t bytes = get_total_piece_size_in_bytes(key.input_shapes) +
                   get_total_piece_size_in_bytes(key.weight_shapes) +
                   get_total_piece_size_in_bytes(key.output_shapes);

    float forward_ms =
        is_parallel_op(key.op_attrs)
            ? parallel_op_runtime_ms(bytes, key.op_attrs)
            : std::max(runtime_ms_from_bytes(bytes), compute_runtime_ms(key));
    return RuntimeOnlyOpCostMetrics{
        /*forward_runtime=*/milliseconds_t{forward_ms},
        /*backward_runtime=*/milliseconds_t{forward_ms * 2.0f},
    };
  }

  milliseconds_t
  estimate_cost(TensorSetMovement const &movement) const override {
    float max_ms = 0.0f;
    for (auto const &[edge, bytes] : movement.edge_to_size) {
      MachineSpaceCoordinate src = edge.get_src();
      MachineSpaceCoordinate dst = edge.get_dst();
      if (src == dst) {
        continue;
      }

      bytes_per_second_t bandwidth =
          (src.node_idx == dst.node_idx)
              ? this->interconnect_specification.intra_node_bandwidth
              : this->interconnect_specification.inter_node_bandwidth;
      max_ms = std::max(max_ms, (bytes / bandwidth).unwrap_milliseconds());
    }
    return milliseconds_t{max_ms};
  }

private:
  static float runtime_ms_from_bytes(size_t bytes) {
    constexpr float assumed_device_bytes_per_second = 3.0e12f;
    float ms =
        static_cast<float>(bytes) / assumed_device_bytes_per_second * 1000.0f;
    return std::max(ms, 0.001f);
  }

  static double tensor_num_elements(TensorShape const &shape) {
    return static_cast<double>(get_num_elements(shape.dims).int_from_positive_int());
  }

  static float runtime_ms_from_flops(double flops,
                                     double flops_per_second) {
    double ms = flops / flops_per_second * 1000.0;
    return static_cast<float>(std::max(ms, 0.001));
  }

  static float runtime_ms_from_linear_flops(double flops) {
    constexpr double assumed_linear_flops_per_second = 30.0e12;
    return runtime_ms_from_flops(flops, assumed_linear_flops_per_second);
  }

  static float runtime_ms_from_attention_flops(double flops) {
    constexpr double assumed_attention_flops_per_second = 120.0e12;
    return runtime_ms_from_flops(flops, assumed_attention_flops_per_second);
  }

  static int parallel_degree(PCGOperatorAttrs const &attrs) {
    if (attrs.has<CombineAttrs>()) {
      return attrs.get<CombineAttrs>().combine_degree.int_from_positive_int();
    }
    if (attrs.has<RepartitionAttrs>()) {
      return attrs.get<RepartitionAttrs>()
          .repartition_degree.int_from_positive_int();
    }
    if (attrs.has<ReplicateAttrs>()) {
      return attrs.get<ReplicateAttrs>()
          .replicate_degree.int_from_positive_int();
    }
    if (attrs.has<ReductionAttrs>()) {
      return attrs.get<ReductionAttrs>()
          .reduction_degree.int_from_positive_int();
    }
    return 1;
  }

  static float parallel_op_runtime_ms(size_t bytes,
                                      PCGOperatorAttrs const &attrs) {
    constexpr float assumed_collective_bytes_per_second = 240.0e12f;
    float degree = static_cast<float>(std::max(parallel_degree(attrs), 1));
    float ms = static_cast<float>(bytes) /
               (assumed_collective_bytes_per_second * degree) * 1000.0f;
    return std::max(ms, 0.00001f);
  }

  static float
      estimate_linear_runtime_ms(RuntimeOnlyOpCostEstimateKey const &key,
                                 LinearAttrs const &) {
    auto input_it = key.input_shapes.find(TensorSlotName::INPUT);
    if (input_it == key.input_shapes.end()) {
      return 0.001f;
    }

    TensorShape input_piece = get_piece_shape(input_it->second);
    double input_channels = static_cast<double>(
        dim_at_idx(input_piece.dims, relative_ff_dim_t{-1})
            .int_from_positive_int());
    double in_channels = input_channels;
    double out_channels = 1.0;
    int tensor_parallel_degree =
        get_total_parallel_degree(input_it->second).int_from_positive_int();

    auto weight_it = key.weight_shapes.find(TensorSlotName::WEIGHT);
    if (weight_it != key.weight_shapes.end()) {
      TensorShape weight_piece = get_piece_shape(weight_it->second);
      tensor_parallel_degree =
          std::max(tensor_parallel_degree,
                   get_total_parallel_degree(weight_it->second)
                       .int_from_positive_int());
      out_channels = static_cast<double>(
          dim_at_idx(weight_piece.dims, relative_ff_dim_t{0})
              .int_from_positive_int());
      in_channels = static_cast<double>(
          dim_at_idx(weight_piece.dims, relative_ff_dim_t{-1})
              .int_from_positive_int());
    } else {
      auto output_it = key.output_shapes.find(TensorSlotName::OUTPUT);
      if (output_it == key.output_shapes.end()) {
        return 0.001f;
      }
      TensorShape output_piece = get_piece_shape(output_it->second);
      tensor_parallel_degree =
          std::max(tensor_parallel_degree,
                   get_total_parallel_degree(output_it->second)
                       .int_from_positive_int());
      out_channels = static_cast<double>(
          dim_at_idx(output_piece.dims, relative_ff_dim_t{-1})
              .int_from_positive_int());
    }

    auto output_it = key.output_shapes.find(TensorSlotName::OUTPUT);
    if (output_it != key.output_shapes.end()) {
      tensor_parallel_degree =
          std::max(tensor_parallel_degree,
                   get_total_parallel_degree(output_it->second)
                       .int_from_positive_int());
    }

    double batch_elements = tensor_num_elements(input_piece) / input_channels;
    double flops = 2.0 * batch_elements * in_channels * out_channels /
                   static_cast<double>(std::max(tensor_parallel_degree, 1));
    return runtime_ms_from_linear_flops(flops);
  }

  static float estimate_attention_runtime_ms(
      RuntimeOnlyOpCostEstimateKey const &key,
      MultiHeadAttentionAttrs const &attrs) {
    auto query_it = key.input_shapes.find(TensorSlotName::QUERY);
    auto output_it = key.output_shapes.find(TensorSlotName::OUTPUT);
    if (query_it == key.input_shapes.end() ||
        output_it == key.output_shapes.end()) {
      return 0.001f;
    }

    TensorShape query_piece = get_piece_shape(query_it->second);
    TensorShape output_piece = get_piece_shape(output_it->second);
    double batch = static_cast<double>(
        dim_at_idx(query_piece.dims, relative_ff_dim_t{-3})
            .int_from_positive_int());
    double seq_len = static_cast<double>(
        dim_at_idx(query_piece.dims, relative_ff_dim_t{-2})
            .int_from_positive_int());
    double embed_dim = static_cast<double>(
        dim_at_idx(output_piece.dims, relative_ff_dim_t{-1})
            .int_from_positive_int());
    double heads =
        static_cast<double>(attrs.num_heads.int_from_positive_int());
    double head_dim = embed_dim / heads;
    int tensor_parallel_degree =
        std::max(get_total_parallel_degree(query_it->second).int_from_positive_int(),
                 get_total_parallel_degree(output_it->second).int_from_positive_int());
    for (auto const &[_, weight_shape] : key.weight_shapes) {
      tensor_parallel_degree =
          std::max(tensor_parallel_degree,
                   get_total_parallel_degree(weight_shape)
                       .int_from_positive_int());
    }

    double projection_flops = 8.0 * batch * seq_len * embed_dim * embed_dim;
    double attention_flops =
        4.0 * batch * heads * seq_len * seq_len * head_dim;
    double flops = (projection_flops + attention_flops) /
                   static_cast<double>(std::max(tensor_parallel_degree, 1));
    return runtime_ms_from_attention_flops(flops);
  }

  static float compute_runtime_ms(RuntimeOnlyOpCostEstimateKey const &key) {
    if (key.op_attrs.has<LinearAttrs>()) {
      return estimate_linear_runtime_ms(key, key.op_attrs.get<LinearAttrs>());
    }
    if (key.op_attrs.has<MultiHeadAttentionAttrs>()) {
      return estimate_attention_runtime_ms(
          key, key.op_attrs.get<MultiHeadAttentionAttrs>());
    }
    return 0.001f;
  }

private:
  MachineInterconnectSpecification interconnect_specification;
};

RuntimeOnlyCostEstimator make_heuristic_runtime_only_cost_estimator(
    MachineInterconnectSpecification const &interconnect_specification) {
  return RuntimeOnlyCostEstimator::create<HeuristicRuntimeOnlyCostEstimator>(
      interconnect_specification);
}

template <typename EventMap> void wait_for_events(EventMap const &events) {
  using Event = std::decay_t<decltype(events.begin()->second)>;

  std::vector<Event> event_values;
  event_values.reserve(events.size());
  for (auto const &kv : events) {
    event_values.push_back(kv.second);
  }
  Event::merge_events(event_values).wait();
}

int count_realm_processors(Realm::Processor::Kind kind) {
  int result = 0;
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  for (Realm::Processor proc : pq) {
    if (proc.kind() == kind) {
      result++;
    }
  }
  return result;
}

struct SampleParallelizationStats {
  size_t sample_parallelized_layers = 0;
  int max_sample_parallel_degree = 1;
};

bool tensor_has_sample_parallelism(ParallelTensorShape const &shape) {
  if (num_shard_dims(shape).value.unwrap_nonnegative() <= 0) {
    return false;
  }
  return shard_dim_at_idx(shape, relative_ff_dim_t{0}).degree > 1_p;
}

SampleParallelizationStats get_sample_parallelization_stats_from_pcg(
    ParallelComputationGraph const &pcg) {
  SampleParallelizationStats result;
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    bool layer_is_sample_parallel = false;
    auto visit_tensor = [&](parallel_tensor_guid_t tensor) {
      ParallelTensorShape shape = get_parallel_tensor_shape(pcg, tensor);
      if (tensor_has_sample_parallelism(shape)) {
        layer_is_sample_parallel = true;
        result.max_sample_parallel_degree =
            std::max(result.max_sample_parallel_degree,
                     shard_dim_at_idx(shape, relative_ff_dim_t{0})
                         .degree.int_from_positive_int());
      }
    };

    for (auto const &[_, tensor] : get_incoming_tensors(pcg, layer)) {
      visit_tensor(tensor);
    }
    for (auto const &[_, tensor] : get_outgoing_tensors(pcg, layer)) {
      visit_tensor(tensor);
    }

    if (layer_is_sample_parallel) {
      result.sample_parallelized_layers++;
    }
  }
  return result;
}

void print_mapping_summary(ParallelComputationGraph const &pcg,
                           MachineMapping const &mapping) {
  SampleParallelizationStats sample_stats =
      get_sample_parallelization_stats_from_pcg(pcg);
  std::unordered_set<MachineSpaceCoordinate> touched_devices;
  std::unordered_map<MachineSpaceCoordinate, size_t> layers_by_device;
  std::unordered_map<MachineSpaceCoordinate, size_t> mha_layers_by_device;
  size_t num_layers = 0;
  size_t num_mha_layers = 0;
  size_t num_parallel_ops = 0;
  size_t num_multi_device_layers = 0;
  size_t num_task_device_bindings = 0;

  for (parallel_layer_guid_t layer : topological_ordering(pcg)) {
    OperatorTaskSpace task = get_operator_task_space(pcg, layer);
    MachineView machine_view = mapping.machine_views.at(layer);
    std::unordered_set<MachineSpaceCoordinate> device_coords =
        get_machine_space_coordinates(task, machine_view);
    bool is_mha = pcg_get_op_attrs(pcg, layer).has<MultiHeadAttentionAttrs>();
    bool is_parallel = is_parallel_op(pcg_get_op_attrs(pcg, layer));

    num_layers++;
    if (is_mha) {
      num_mha_layers++;
    }
    if (is_parallel) {
      num_parallel_ops++;
    }
    if (device_coords.size() > 1) {
      num_multi_device_layers++;
    }
    num_task_device_bindings += device_coords.size();
    for (MachineSpaceCoordinate const &device_coord : device_coords) {
      touched_devices.insert(device_coord);
      layers_by_device[device_coord]++;
      if (is_mha) {
        mha_layers_by_device[device_coord]++;
      }
    }
  }

  std::cerr << "[llama2-benchmark] mapping summary: layers=" << num_layers
            << ", mha_layers=" << num_mha_layers
            << ", parallel_ops=" << num_parallel_ops
            << ", multi_device_layers=" << num_multi_device_layers
            << ", touched_devices=" << touched_devices.size()
            << ", task_device_bindings=" << num_task_device_bindings
            << ", sample_parallelized_layers="
            << sample_stats.sample_parallelized_layers
            << ", max_sample_parallel_degree="
            << sample_stats.max_sample_parallel_degree << "\n";
  for (auto const &[device_coord, num_device_layers] : layers_by_device) {
    std::cerr << "[llama2-benchmark] mapping device=" << device_coord
              << ", layer_bindings=" << num_device_layers << "\n";
  }
  for (auto const &[device_coord, num_device_mha_layers] :
       mha_layers_by_device) {
    std::cerr << "[llama2-benchmark] mapping device=" << device_coord
              << ", mha_layer_bindings=" << num_device_mha_layers << "\n";
  }
}

int max_single_machine_view_data_parallel_degree(BenchmarkArgs const &args) {
  return std::max(args.num_nodes, args.num_gpus);
}

bool requires_flattened_data_parallel_mapping(BenchmarkArgs const &args) {
  return args.data_parallel_degree > 1;
}

MachineSpaceCoordinate
    flattened_gpu_coord_for_index(BenchmarkArgs const &args, int index) {
  int total_gpus = total_num_gpus(args);
  int flat = index % total_gpus;
  return MachineSpaceCoordinate{
      nonnegative_int{flat / args.num_gpus},
      nonnegative_int{flat % args.num_gpus},
      DeviceType::GPU,
  };
}

int flattened_task_index(OperatorTaskSpace const &task_space,
                         TaskSpaceCoordinate const &coord) {
  int flat = 0;
  int stride = 1;
  for (size_t idx = 0; idx < coord.orthotope_coord.raw.size(); idx++) {
    flat += coord.orthotope_coord.raw.at(idx).unwrap_nonnegative() * stride;
    stride *= task_space.degrees.dims.at(idx)
                  .positive_int_from_int_ge_two()
                  .int_from_positive_int();
  }
  return flat;
}

int stable_flattened_task_index(BenchmarkArgs const &args,
                                OperatorTaskSpace const &task_space,
                                TaskSpaceCoordinate const &coord,
                                int layer_ordinal) {
  auto dim_degree = [&](size_t idx) {
    return task_space.degrees.dims.at(idx)
        .positive_int_from_int_ge_two()
        .int_from_positive_int();
  };

  if (task_space.degrees.dims.size() == 1 &&
      dim_degree(0) == args.data_parallel_degree &&
      args.num_nodes >= args.data_parallel_degree) {
    int sample_idx = coord.orthotope_coord.raw.at(0).unwrap_nonnegative();
    int local_offset = layer_ordinal % args.num_gpus;
    return sample_idx * args.num_gpus + local_offset;
  }

  if (task_space.degrees.dims.size() == 2 &&
      args.num_nodes >= args.data_parallel_degree) {
    int first_degree = dim_degree(0);
    int second_degree = dim_degree(1);
    int first_idx = coord.orthotope_coord.raw.at(0).unwrap_nonnegative();
    int second_idx = coord.orthotope_coord.raw.at(1).unwrap_nonnegative();

    if (first_degree == args.data_parallel_degree &&
        second_degree <= args.num_gpus) {
      int group_count = std::max(1, args.num_gpus / second_degree);
      int local_offset = (layer_ordinal % group_count) * second_degree;
      return first_idx * args.num_gpus + local_offset + second_idx;
    }
    if (second_degree == args.data_parallel_degree &&
        first_degree <= args.num_gpus) {
      int group_count = std::max(1, args.num_gpus / first_degree);
      int local_offset = (layer_ordinal % group_count) * first_degree;
      return second_idx * args.num_gpus + local_offset + first_idx;
    }
  }

  return flattened_task_index(task_space, coord);
}

nonnegative_int clamp_parallel_coord_component(nonnegative_int coord,
                                               positive_int degree) {
  int raw_coord = coord.unwrap_nonnegative();
  int raw_degree = degree.int_from_positive_int();
  return nonnegative_int{raw_degree <= 1 ? 0 : raw_coord % raw_degree};
}

ParallelTensorSpaceCoordinate project_parallel_coord_to_degrees(
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

struct TensorCoordPlan {
  TensorSlotName slot_name;
  OperatorSpaceToParallelTensorSpaceMapping mapping;
  num_ptensor_shard_dims_t num_shard_dims;
  std::optional<ParallelTensorDimDegrees> projection_degrees = std::nullopt;
};

bool direct_mapping_should_use_output_identity_task_space(
    PCGOperatorAttrs const &pcg_op_attrs) {
  return is_parallel_op(pcg_op_attrs) || pcg_op_attrs.has<WeightAttrs>();
}

OperatorTaskSpace direct_mapping_task_space_for_layer(
    ParallelComputationGraph const &pcg, parallel_layer_guid_t layer) {
  PCGOperatorAttrs pcg_op_attrs = pcg_get_op_attrs(pcg, layer);
  if (direct_mapping_should_use_output_identity_task_space(pcg_op_attrs)) {
    parallel_tensor_guid_t output_tensor =
        get_outgoing_tensors(pcg, layer).at(TensorSlotName::OUTPUT);
    return get_operator_task_space_matching_parallel_tensor_dim_degrees(
        get_parallel_degrees(get_parallel_tensor_shape(pcg, output_tensor)));
  }

  return get_operator_task_space(pcg, layer);
}

std::vector<TensorCoordPlan>
    make_tensor_coord_plans_for_layer(ParallelComputationGraph const &pcg,
                                      parallel_layer_guid_t layer) {
  PCGOperatorAttrs pcg_op_attrs = pcg_get_op_attrs(pcg, layer);

  if (direct_mapping_should_use_output_identity_task_space(pcg_op_attrs)) {
    OperatorTaskSpace task_space =
        direct_mapping_task_space_for_layer(pcg, layer);
    parallel_tensor_guid_t output_tensor =
        get_outgoing_tensors(pcg, layer).at(TensorSlotName::OUTPUT);
    ParallelTensorDimDegrees output_degrees =
        get_parallel_degrees(get_parallel_tensor_shape(pcg, output_tensor));
    OperatorSpaceToParallelTensorSpaceMapping output_mapping =
        get_identity_mapping(task_space, output_degrees);
    num_ptensor_shard_dims_t num_shard_dims =
        get_ptensor_dim_degrees_num_shard_dims(output_degrees);
    bool project_input_slots = pcg_op_attrs.has<CombineAttrs>() ||
                               pcg_op_attrs.has<ReductionAttrs>();

    std::unordered_set<TensorSlotName> slot_names =
        keys(get_outgoing_tensors(pcg, layer));
    for (TensorSlotName const &slot_name :
         keys(get_incoming_tensors(pcg, layer))) {
      slot_names.insert(slot_name);
    }

    std::vector<TensorCoordPlan> plans;
    for (TensorSlotName const &slot_name : slot_names) {
      parallel_tensor_guid_t slot_tensor =
          contains_key(get_outgoing_tensors(pcg, layer), slot_name)
              ? get_outgoing_tensors(pcg, layer).at(slot_name)
              : get_incoming_tensors(pcg, layer).at(slot_name);
      ParallelTensorDimDegrees slot_degrees =
          get_parallel_degrees(get_parallel_tensor_shape(pcg, slot_tensor));
      plans.push_back(TensorCoordPlan{
          /*slot_name=*/slot_name,
          /*mapping=*/output_mapping,
          /*num_shard_dims=*/num_shard_dims,
          /*projection_degrees=*/
          (project_input_slots || slot_name == TensorSlotName::OUTPUT)
              ? std::optional<ParallelTensorDimDegrees>{slot_degrees}
              : std::nullopt,
      });
    }
    return plans;
  }

  ComputationGraphOpAttrs op_attrs =
      compgraph_op_attrs_from_pcg_op_attrs(pcg_op_attrs).value();
  std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>
      inputs_dim_degrees = get_incoming_input_degrees(pcg, layer);
  std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
      mappings = get_operator_to_ptensor_mappings(op_attrs, inputs_dim_degrees);

  std::vector<TensorCoordPlan> plans;
  for (auto const &[slot_name, mapping] : mappings) {
    ParallelTensorDimDegrees tensor_degrees =
        get_parallel_tensor_space_for_mapping(mapping);
    num_ptensor_shard_dims_t num_shard_dims =
        get_ptensor_dim_degrees_num_shard_dims(tensor_degrees);
    plans.push_back(TensorCoordPlan{
        /*slot_name=*/slot_name,
        /*mapping=*/mapping,
        /*num_shard_dims=*/num_shard_dims,
        /*projection_degrees=*/std::nullopt,
    });
  }

  return plans;
}

OperatorAtomicTaskShardBinding make_task_binding_for_task_coord(
    std::vector<TensorCoordPlan> const &plans,
    TaskSpaceCoordinate const &task_coord) {
  std::unordered_map<TensorSlotName, ParallelTensorSpaceCoordinate>
      tensor_coords;
  for (TensorCoordPlan const &plan : plans) {
    ParallelTensorSpaceCoordinate tensor_coord =
        ptensor_coord_for_task_space_coord(plan.mapping,
                                           task_coord,
                                           plan.num_shard_dims);
    if (plan.projection_degrees.has_value()) {
      tensor_coord = project_parallel_coord_to_degrees(
          tensor_coord, plan.projection_degrees.value());
    }
    tensor_coords.insert({plan.slot_name, tensor_coord});
  }
  return OperatorAtomicTaskShardBinding{/*tensor_coords=*/tensor_coords};
}

MappedOperatorTaskGroup create_flattened_data_parallel_task_group(
    ParallelComputationGraph const &pcg, parallel_layer_guid_t layer,
    BenchmarkArgs const &args, int layer_ordinal) {
  bool log_layer_detail = layer_ordinal < 5 || layer_ordinal % 100 == 0;
  ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(pcg, layer);
  if (log_layer_detail) {
    std::cerr << "[llama2-benchmark] direct mapping layer begin"
              << ": ordinal=" << layer_ordinal
              << ", name=" << layer_attrs.name.value_or("<unnamed>")
              << ", op_type="
              << pcg_op_attrs_get_op_type(layer_attrs.op_attrs) << "\n";
  }
  OperatorTaskSpace task_space = direct_mapping_task_space_for_layer(pcg, layer);
  if (log_layer_detail) {
    std::cerr << "[llama2-benchmark] direct mapping task space"
              << ": ordinal=" << layer_ordinal
              << ", num_tasks=" << num_tasks(task_space)
              << ", task_space=" << task_space << "\n";
  }
  std::unordered_set<TaskSpaceCoordinate> task_coords =
      get_task_space_coordinates(task_space);
  if (log_layer_detail) {
    std::cerr << "[llama2-benchmark] direct mapping task coords"
              << ": ordinal=" << layer_ordinal
              << ", coords=" << task_coords.size() << "\n";
  }
  bool is_single_task = task_coords.size() == 1;
  std::vector<TensorCoordPlan> tensor_coord_plans =
      make_tensor_coord_plans_for_layer(pcg, layer);
  if (log_layer_detail) {
    std::cerr << "[llama2-benchmark] direct mapping tensor plans"
              << ": ordinal=" << layer_ordinal
              << ", plans=" << tensor_coord_plans.size() << "\n";
  }

  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> bindings;
  for (TaskSpaceCoordinate const &task_coord : task_coords) {
    int flat_index =
        is_single_task ? layer_ordinal
                       : stable_flattened_task_index(args,
                                                     task_space,
                                                     task_coord,
                                                     layer_ordinal);
    bindings.equate(flattened_gpu_coord_for_index(args, flat_index),
                    make_task_binding_for_task_coord(tensor_coord_plans,
                                                     task_coord));
  }
  if (log_layer_detail) {
    std::cerr << "[llama2-benchmark] direct mapping layer done"
              << ": ordinal=" << layer_ordinal
              << ", bindings=" << bindings.size() << "\n";
  }

  return MappedOperatorTaskGroup{bindings};
}

MappedParallelComputationGraph create_flattened_data_parallel_mpcg(
    ParallelComputationGraph const &pcg, BenchmarkArgs const &args) {
  std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup>
      mapped_groups;
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  std::cerr << "[llama2-benchmark] direct mapping task-group build start"
            << ": layers=" << layers.size() << "\n";
  for (size_t idx = 0; idx < layers.size(); idx++) {
    if (idx % 100 == 0) {
      std::cerr << "[llama2-benchmark] direct mapping progress: layer "
                << idx << "/" << layers.size() << "\n";
    }
    parallel_layer_guid_t layer = layers.at(idx);
    mapped_groups.emplace(layer,
                          create_flattened_data_parallel_task_group(
                              pcg, layer, args, static_cast<int>(idx)));
  }
  std::cerr << "[llama2-benchmark] direct mapping task-group build done"
            << ": groups=" << mapped_groups.size() << "\n";

  std::cerr << "[llama2-benchmark] direct mapping graph assembly start\n";
  MappedParallelComputationGraph result =
      mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg, mapped_groups);
  std::cerr << "[llama2-benchmark] direct mapping graph assembly done\n";
  return result;
}

void print_mapped_pcg_summary(ParallelComputationGraph const &pcg,
                              MappedParallelComputationGraph const &mpcg) {
  SampleParallelizationStats sample_stats =
      get_sample_parallelization_stats_from_pcg(pcg);
  std::unordered_set<MachineSpaceCoordinate> touched_devices;
  std::unordered_map<MachineSpaceCoordinate, size_t> layers_by_device;
  std::unordered_map<MachineSpaceCoordinate, size_t> mha_layers_by_device;
  size_t num_layers = 0;
  size_t num_mha_layers = 0;
  size_t num_parallel_ops = 0;
  size_t num_multi_device_layers = 0;
  size_t num_task_device_bindings = 0;

  for (parallel_layer_guid_t layer : topological_ordering(pcg)) {
    MappedOperatorTaskGroup task_group = mpcg_get_mapping_for_layer(mpcg,
                                                                    layer);
    std::unordered_set<MachineSpaceCoordinate> device_coords =
        task_group.get_shard_bindings().left_values();
    bool is_mha = pcg_get_op_attrs(pcg, layer).has<MultiHeadAttentionAttrs>();
    bool is_parallel = is_parallel_op(pcg_get_op_attrs(pcg, layer));

    num_layers++;
    if (is_mha) {
      num_mha_layers++;
    }
    if (is_parallel) {
      num_parallel_ops++;
    }
    if (device_coords.size() > 1) {
      num_multi_device_layers++;
    }
    num_task_device_bindings += device_coords.size();
    for (MachineSpaceCoordinate const &device_coord : device_coords) {
      touched_devices.insert(device_coord);
      layers_by_device[device_coord]++;
      if (is_mha) {
        mha_layers_by_device[device_coord]++;
      }
    }
  }

  std::cerr << "[llama2-benchmark] mapped PCG summary: layers=" << num_layers
            << ", mha_layers=" << num_mha_layers
            << ", parallel_ops=" << num_parallel_ops
            << ", multi_device_layers=" << num_multi_device_layers
            << ", touched_devices=" << touched_devices.size()
            << ", task_device_bindings=" << num_task_device_bindings
            << ", sample_parallelized_layers="
            << sample_stats.sample_parallelized_layers
            << ", max_sample_parallel_degree="
            << sample_stats.max_sample_parallel_degree << "\n";
  for (auto const &[device_coord, num_device_layers] : layers_by_device) {
    std::cerr << "[llama2-benchmark] mapped device=" << device_coord
              << ", layer_bindings=" << num_device_layers << "\n";
  }
  for (auto const &[device_coord, num_device_mha_layers] :
       mha_layers_by_device) {
    std::cerr << "[llama2-benchmark] mapped device=" << device_coord
              << ", mha_layer_bindings=" << num_device_mha_layers << "\n";
  }
}

SearchResult create_balanced_mapping(
    ParallelComputationGraph const &pcg,
    MachineComputeSpecification const &machine_compute_spec) {
  std::optional<MachineMapping> maybe_mapping =
      get_random_mapping(pcg, machine_compute_spec, DeviceType::GPU);
  if (!maybe_mapping.has_value()) {
    throw std::runtime_error("failed to create a balanced GPU mapping for "
                             "llama2_7b_like");
  }

  return SearchResult{
      /*pcg=*/pcg,
      /*machine_mapping=*/maybe_mapping.value(),
  };
}

void assert_reusable_layer_attrs_match(
    ParallelComputationGraph const &full_pcg, parallel_layer_guid_t full_layer,
    ParallelComputationGraph const &searched_pcg,
    parallel_layer_guid_t searched_layer) {
  ParallelLayerAttrs full_attrs =
      get_parallel_layer_attrs(full_pcg, full_layer);
  ParallelLayerAttrs searched_attrs =
      get_parallel_layer_attrs(searched_pcg, searched_layer);
  OperatorType full_op_type = pcg_op_attrs_get_op_type(full_attrs.op_attrs);
  OperatorType searched_op_type =
      pcg_op_attrs_get_op_type(searched_attrs.op_attrs);
  OperatorTaskSpace full_task_space = get_operator_task_space(full_pcg, full_layer);
  OperatorTaskSpace searched_task_space =
      get_operator_task_space(searched_pcg, searched_layer);
  if (full_op_type != searched_op_type ||
      full_task_space != searched_task_space) {
    std::ostringstream oss;
    oss << "cannot reuse decoder-layer mapping because searched and full PCG "
           "layer attributes differ: full_layer="
        << full_layer << ", searched_layer=" << searched_layer
        << ", full_name=" << full_attrs.name.value_or("<unnamed>")
        << ", searched_name=" << searched_attrs.name.value_or("<unnamed>")
        << ", full_op_type=" << full_op_type
        << ", searched_op_type=" << searched_op_type
        << ", full_task_space=" << full_task_space
        << ", searched_task_space=" << searched_task_space;
    throw std::runtime_error(oss.str());
  }
}

std::vector<parallel_layer_guid_t>
construction_ordering(ParallelComputationGraph const &pcg) {
  std::unordered_set<parallel_layer_guid_t> layer_set =
      get_parallel_layers(pcg);
  std::vector<parallel_layer_guid_t> layers(layer_set.begin(), layer_set.end());
  std::sort(layers.begin(), layers.end());
  return layers;
}

struct LinearParallelizationStats {
  size_t original_linear_layers = 0;
  size_t transformed_linear_layers = 0;
  size_t pcg_layers_before = 0;
  size_t pcg_layers_after = 0;
};

struct LinearParallelizationResult {
  ParallelComputationGraph pcg;
  LinearParallelizationStats stats;
};

struct AttentionParallelizationStats {
  size_t original_attention_layers = 0;
  size_t transformed_attention_layers = 0;
  size_t pcg_layers_before = 0;
  size_t pcg_layers_after = 0;
};

struct AttentionParallelizationResult {
  ParallelComputationGraph pcg;
  AttentionParallelizationStats stats;
};

size_t count_linear_parallelized_layers_from_pcg(
    ParallelComputationGraph const &pcg) {
  size_t result = 0;
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    if (pcg_get_op_attrs(pcg, layer).has<CombineAttrs>()) {
      result++;
    }
  }
  return result;
}

size_t count_attention_parallelized_layers_from_pcg(
    ParallelComputationGraph const &pcg) {
  size_t result = 0;
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    if (pcg_get_op_attrs(pcg, layer).has<ReductionAttrs>()) {
      result++;
    }
  }
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

int get_max_reduction_parallel_degree_from_pcg(
    ParallelComputationGraph const &pcg) {
  int result = 1;
  for (parallel_layer_guid_t layer : get_parallel_layers(pcg)) {
    PCGOperatorAttrs attrs = pcg_get_op_attrs(pcg, layer);
    if (attrs.has<ReductionAttrs>()) {
      result = std::max(
          result,
          attrs.get<ReductionAttrs>().reduction_degree.int_from_positive_int());
    }
  }
  return result;
}

int get_max_parallel_collective_degree_from_pcg(
    ParallelComputationGraph const &pcg) {
  return std::max(get_max_combine_parallel_degree_from_pcg(pcg),
                  get_max_reduction_parallel_degree_from_pcg(pcg));
}

bool layer_exists(ParallelComputationGraph const &pcg,
                  parallel_layer_guid_t layer) {
  return get_parallel_layers(pcg).count(layer) > 0;
}

std::vector<parallel_layer_guid_t>
get_linear_layers_in_construction_order(ParallelComputationGraph const &pcg) {
  std::vector<parallel_layer_guid_t> result;
  for (parallel_layer_guid_t layer : construction_ordering(pcg)) {
    if (pcg_get_op_attrs(pcg, layer).has<LinearAttrs>()) {
      result.push_back(layer);
    }
  }
  return result;
}

std::vector<parallel_layer_guid_t>
get_attention_layers_in_construction_order(ParallelComputationGraph const &pcg) {
  std::vector<parallel_layer_guid_t> result;
  for (parallel_layer_guid_t layer : construction_ordering(pcg)) {
    if (pcg_get_op_attrs(pcg, layer).has<MultiHeadAttentionAttrs>()) {
      result.push_back(layer);
    }
  }
  return result;
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

std::optional<PCGPatternMatch>
find_match_touching_layer(ParallelComputationGraph const &pcg,
                          Substitution const &substitution,
                          parallel_layer_guid_t target_layer) {
  SubParallelComputationGraph sub_pcg = sub_pcg_from_full_pcg(pcg);
  for (PCGPatternMatch const &match :
       find_pattern_matches(substitution.pcg_pattern, sub_pcg)) {
    if (right_entries(match.node_assignment).count(target_layer) > 0) {
      return match;
    }
  }
  return std::nullopt;
}

ff_dim_t last_shard_dim(ParallelTensorShape const &shape) {
  int rank = num_shard_dims(shape.dims).value.unwrap_nonnegative();
  if (rank <= 0) {
    throw std::runtime_error("cannot get last shard dim of scalar tensor");
  }
  return ff_dim_t{nonnegative_int{rank - 1}};
}

parallel_tensor_guid_t create_parallelized_linear(
    ParallelComputationGraphBuilder &pcgb, parallel_tensor_guid_t const &input,
    positive_int out_dim, std::optional<Activation> activation, bool use_bias,
    DataType data_type, int degree, std::optional<std::string> const &name) {
  if (degree <= 1) {
    return pcgb.dense(input,
                      out_dim,
                      activation,
                      use_bias,
                      data_type,
                      /*projection_initializer=*/std::nullopt,
                      /*bias_initializer=*/std::nullopt,
                      name);
  }

  positive_int parallel_degree{degree};
  std::string base_name = name.value_or("linear");
  parallel_tensor_guid_t replicated_input =
      pcgb.parallel_replicate(input, parallel_degree, base_name + "_replicate");
  parallel_tensor_guid_t projected =
      pcgb.dense(replicated_input,
                 out_dim,
                 activation,
                 use_bias,
                 data_type,
                 /*projection_initializer=*/std::nullopt,
                 /*bias_initializer=*/std::nullopt,
                 name);
  return pcgb.parallel_combine(projected,
                               last_shard_dim(pcgb.get_shape(projected)),
                               parallel_degree,
                               base_name + "_combine");
}

parallel_tensor_guid_t create_parallelized_llama2_feedforward_network(
    ParallelComputationGraphBuilder &pcgb, TransformerConfig const &config,
    parallel_tensor_guid_t const &input, int linear_parallel_degree) {
  parallel_tensor_guid_t gate =
      create_parallelized_linear(pcgb,
                                 input,
                                 config.dim_feedforward,
                                 Activation::GELU,
                                 /*use_bias=*/false,
                                 DataType::FLOAT,
                                 linear_parallel_degree,
                                 "gate_proj");
  parallel_tensor_guid_t up =
      create_parallelized_linear(pcgb,
                                 input,
                                 config.dim_feedforward,
                                 /*activation=*/std::nullopt,
                                 /*use_bias=*/false,
                                 DataType::FLOAT,
                                 linear_parallel_degree,
                                 "up_proj");
  parallel_tensor_guid_t gated = pcgb.multiply(gate, up, "ffn_gated");
  return create_parallelized_linear(pcgb,
                                    gated,
                                    config.num_features,
                                    /*activation=*/std::nullopt,
                                    /*use_bias=*/false,
                                    DataType::FLOAT,
                                    linear_parallel_degree,
                                    "down_proj");
}

parallel_tensor_guid_t create_parallelized_llama2_self_attention(
    ParallelComputationGraphBuilder &pcgb, TransformerConfig const &config,
    parallel_tensor_guid_t const &input, int attention_parallel_degree) {
  if (attention_parallel_degree <= 1) {
    return pcgb.multihead_attention(/*query=*/input,
                                    /*key=*/input,
                                    /*value=*/input,
                                    /*embed_dim=*/config.num_features,
                                    /*num_heads=*/config.num_heads,
                                    /*kdim=*/config.num_features,
                                    /*vdim=*/config.num_features,
                                    /*dropout=*/config.dropout,
                                    /*bias=*/false,
                                    /*add_bias_kv=*/false,
                                    /*add_zero_attn=*/false,
                                    /*initializer=*/std::nullopt,
                                    /*input_bias_initializer=*/std::nullopt,
                                    /*output_bias_initializer=*/std::nullopt,
                                    /*name=*/"self_attention");
  }

  positive_int parallel_degree{attention_parallel_degree};
  parallel_tensor_guid_t replicated_input = pcgb.parallel_replicate(
      input, parallel_degree, "self_attention_replicate");
  parallel_tensor_guid_t attention =
      pcgb.multihead_attention(/*query=*/replicated_input,
                               /*key=*/replicated_input,
                               /*value=*/replicated_input,
                               /*embed_dim=*/config.num_features,
                               /*num_heads=*/config.num_heads,
                               /*kdim=*/config.num_features,
                               /*vdim=*/config.num_features,
                               /*dropout=*/config.dropout,
                               /*bias=*/false,
                               /*add_bias_kv=*/false,
                               /*add_zero_attn=*/false,
                               /*initializer=*/std::nullopt,
                               /*input_bias_initializer=*/std::nullopt,
                               /*output_bias_initializer=*/std::nullopt,
                               /*name=*/"self_attention");
  return pcgb.parallel_reduce(attention,
                              parallel_degree,
                              "self_attention_reduce");
}

parallel_tensor_guid_t create_parallelized_llama2_decoder_layer(
    ParallelComputationGraphBuilder &pcgb, TransformerConfig const &config,
    parallel_tensor_guid_t const &input, int linear_parallel_degree,
    int attention_parallel_degree) {
  std::set<relative_ff_dim_t> layer_norm_axis = {relative_ff_dim_t{-1}};
  parallel_tensor_guid_t attention_input =
      pcgb.layer_norm(input,
                      layer_norm_axis,
                      /*elementwise_affine=*/true,
                      config.layer_norm_eps,
                      "input_layernorm");
  parallel_tensor_guid_t self_attention =
      create_parallelized_llama2_self_attention(
          pcgb, config, attention_input, attention_parallel_degree);
  parallel_tensor_guid_t attention_residual =
      pcgb.add(input, self_attention, "attention_residual");
  parallel_tensor_guid_t feedforward_input =
      pcgb.layer_norm(attention_residual,
                      layer_norm_axis,
                      /*elementwise_affine=*/true,
                      config.layer_norm_eps,
                      "post_attention_layernorm");
  parallel_tensor_guid_t feedforward_output =
      create_parallelized_llama2_feedforward_network(
          pcgb, config, feedforward_input, linear_parallel_degree);
  return pcgb.add(attention_residual, feedforward_output, "decoder_output");
}

ParallelComputationGraph create_parallelized_llama2_benchmark_pcg(
    TransformerConfig const &config, int linear_parallel_degree,
    int data_parallel_degree, int attention_parallel_degree = 1) {
  ParallelComputationGraphBuilder pcgb;
  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          config.batch_size, config.sequence_length, config.num_features}},
      DataType::FLOAT,
  };
  parallel_tensor_guid_t hidden_states =
      pcgb.create_input_tensor(input_shape, "input");
  if (data_parallel_degree > 1) {
    hidden_states =
        pcgb.parallel_partition(hidden_states,
                                ff_dim_t{0_n},
                                positive_int{data_parallel_degree},
                                "data_parallel_partition");
  }

  for (int i = 0; i < config.num_decoder_layers.int_from_positive_int(); i++) {
    hidden_states = create_parallelized_llama2_decoder_layer(
        pcgb,
        config,
        hidden_states,
        linear_parallel_degree,
        attention_parallel_degree);
  }

  create_parallelized_linear(pcgb,
                             hidden_states,
                             config.vocab_size,
                             /*activation=*/std::nullopt,
                             /*use_bias=*/false,
                             DataType::FLOAT,
                             linear_parallel_degree,
                             "lm_head");
  return pcgb.pcg;
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
  PCGPatternMatch match{
      /*node_assignment=*/node_assignment,
      /*input_assignment=*/input_assignment,
  };

  if (validate_assignment) {
    SubParallelComputationGraph sub_pcg = sub_pcg_from_full_pcg(pcg);
    if (!assignment_satisfies(sub_pcg, substitution.pcg_pattern, match)) {
      return std::nullopt;
    }
  }

  return match;
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
  size_t rank = static_cast<size_t>(
      num_shard_dims(output_shape.dims).value.unwrap_nonnegative());
  if (rank == 0 || rank > static_cast<size_t>(MAX_TENSOR_DIM)) {
    return std::nullopt;
  }

  return positive_int{static_cast<int>(rank)};
}

void log_linear_parallelization_candidates(ParallelComputationGraph const &pcg,
                                           int degree) {
  std::cerr << "[llama2-benchmark] Linear parallelization candidates:\n";
  for (parallel_layer_guid_t layer :
       get_linear_layers_in_construction_order(pcg)) {
    LinearAttrs attrs = pcg_get_op_attrs(pcg, layer).get<LinearAttrs>();
    ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(pcg, layer);
    std::unordered_map<TensorSlotName, parallel_tensor_guid_t> inputs =
        get_incoming_tensors(pcg, layer);
    std::optional<positive_int> maybe_rank =
        get_single_output_tensor_rank(pcg, layer);

    std::cerr << "  layer=" << layer << ", name="
              << (layer_attrs.name.has_value()
                      ? std::string(layer_attrs.name.value())
                      : std::string("<unnamed>"))
              << ", out_channels=" << attrs.out_channels
              << ", use_bias=" << attrs.use_bias << ", out_channels_mod_degree="
              << (attrs.out_channels.int_from_positive_int() % degree)
              << ", output_rank=";
    if (maybe_rank.has_value()) {
      std::cerr << maybe_rank.value();
    } else {
      std::cerr << "<none>";
    }
    std::cerr << ", input_slots=[";
    bool first = true;
    for (auto const &[slot, tensor] : inputs) {
      if (!first) {
        std::cerr << ",";
      }
      first = false;
      std::cerr << format_as(slot);
    }
  std::cerr << "]\n";
  }
}

struct LinearParallelizationCandidate {
  Substitution substitution;
  PCGPatternMatch match;
};

struct AttentionParallelizationCandidate {
  Substitution substitution;
  PCGPatternMatch match;
};

std::optional<LinearParallelizationCandidate>
find_linear_parallel_candidate_for_layer(ParallelComputationGraph const &pcg,
                                         parallel_layer_guid_t layer,
                                         int degree) {
  LinearAttrs attrs = pcg_get_op_attrs(pcg, layer).get<LinearAttrs>();
  std::optional<positive_int> maybe_output_rank =
      get_single_output_tensor_rank(pcg, layer);
  if (!maybe_output_rank.has_value()) {
    return std::nullopt;
  }

  if (attrs.out_channels.int_from_positive_int() % degree != 0) {
    return std::nullopt;
  }

  Substitution substitution = create_replicate_linear_combine_unchecked(
      maybe_output_rank.value(), positive_int{degree}, attrs.use_bias);
  std::optional<PCGPatternMatch> maybe_match =
      make_single_node_pattern_match_for_layer(pcg, substitution, layer);
  if (maybe_match.has_value()) {
    return LinearParallelizationCandidate{
        /*substitution=*/substitution,
        /*match=*/maybe_match.value(),
    };
  }

  return std::nullopt;
}

std::optional<AttentionParallelizationCandidate>
find_attention_parallel_candidate_for_layer(ParallelComputationGraph const &pcg,
                                            parallel_layer_guid_t layer,
                                            int degree) {
  MultiHeadAttentionAttrs attrs =
      pcg_get_op_attrs(pcg, layer).get<MultiHeadAttentionAttrs>();
  if (attention_layer_is_already_parallelized(pcg, layer)) {
    return std::nullopt;
  }

  if (attrs.embed_dim.int_from_positive_int() % degree != 0 ||
      attrs.num_heads.int_from_positive_int() % degree != 0) {
    return std::nullopt;
  }

  Substitution self_attention_substitution =
      create_replicate_self_attention_reduce(positive_int{degree},
                                             positive_int{degree});
  std::optional<PCGPatternMatch> maybe_self_attention_match =
      make_single_node_pattern_match_for_layer(
          pcg, self_attention_substitution, layer,
          /*validate_assignment=*/false);
  if (maybe_self_attention_match.has_value()) {
    return AttentionParallelizationCandidate{
        /*substitution=*/self_attention_substitution,
        /*match=*/maybe_self_attention_match.value(),
    };
  }

  std::vector<Substitution> attention_substitutions = {
      create_replicate_attention_reduce(positive_int{degree},
                                        positive_int{degree}),
  };
  for (Substitution const &substitution : attention_substitutions) {
    std::optional<PCGPatternMatch> maybe_match =
        find_match_touching_layer(pcg, substitution, layer);
    if (maybe_match.has_value()) {
      return AttentionParallelizationCandidate{
          /*substitution=*/substitution,
          /*match=*/maybe_match.value(),
      };
    }
  }

  return std::nullopt;
}

LinearParallelizationResult
apply_linear_parallelization(ParallelComputationGraph pcg, int degree) {
  LinearParallelizationStats stats;
  stats.pcg_layers_before = get_parallel_layers(pcg).size();

  if (degree <= 1) {
    stats.pcg_layers_after = stats.pcg_layers_before;
    return LinearParallelizationResult{/*pcg=*/pcg, /*stats=*/stats};
  }

  std::vector<parallel_layer_guid_t> original_linear_layers =
      get_linear_layers_in_construction_order(pcg);
  stats.original_linear_layers = original_linear_layers.size();

  for (parallel_layer_guid_t layer : original_linear_layers) {
    if (!layer_exists(pcg, layer)) {
      continue;
    }

    std::optional<LinearParallelizationCandidate> maybe_candidate =
        find_linear_parallel_candidate_for_layer(pcg, layer, degree);
    if (!maybe_candidate.has_value()) {
      continue;
    }

    SubParallelComputationGraph sub_pcg =
        apply_substitution(sub_pcg_from_full_pcg(pcg),
                           maybe_candidate->substitution,
                           maybe_candidate->match);
    pcg = pcg_from_sub_pcg_by_dropping_inputs(sub_pcg);
    stats.transformed_linear_layers++;
  }

  stats.pcg_layers_after = get_parallel_layers(pcg).size();
  return LinearParallelizationResult{/*pcg=*/pcg, /*stats=*/stats};
}

AttentionParallelizationResult
apply_attention_parallelization(ParallelComputationGraph pcg, int degree) {
  AttentionParallelizationStats stats;
  stats.pcg_layers_before = get_parallel_layers(pcg).size();

  if (degree <= 1) {
    stats.pcg_layers_after = stats.pcg_layers_before;
    return AttentionParallelizationResult{/*pcg=*/pcg, /*stats=*/stats};
  }

  std::vector<parallel_layer_guid_t> original_attention_layers =
      get_attention_layers_in_construction_order(pcg);
  stats.original_attention_layers = original_attention_layers.size();

  for (parallel_layer_guid_t layer : original_attention_layers) {
    if (!layer_exists(pcg, layer)) {
      continue;
    }

    std::optional<AttentionParallelizationCandidate> maybe_candidate =
        find_attention_parallel_candidate_for_layer(pcg, layer, degree);
    if (!maybe_candidate.has_value()) {
      continue;
    }

    SubParallelComputationGraph sub_pcg =
        apply_substitution(sub_pcg_from_full_pcg(pcg),
                           maybe_candidate->substitution,
                           maybe_candidate->match);
    pcg = pcg_from_sub_pcg_by_dropping_inputs(sub_pcg);
    stats.transformed_attention_layers++;
  }

  stats.pcg_layers_after = get_parallel_layers(pcg).size();
  return AttentionParallelizationResult{/*pcg=*/pcg, /*stats=*/stats};
}

struct Llama2SoapStrategy {
  int data_parallel_degree = 1;
  int linear_parallel_degree = 1;
  int attention_parallel_degree = 1;
  double estimated_cost = std::numeric_limits<double>::infinity();
  double estimated_memory_mib = std::numeric_limits<double>::infinity();
  size_t linear_parallelized_layers = 0;
  size_t attention_parallelized_layers = 0;
};

struct Llama2SoapSearchResult {
  ParallelComputationGraph pcg;
  Llama2SoapStrategy strategy;
  bool used_search = false;
};

std::vector<int> power_of_two_degrees_up_to(int max_degree) {
  std::vector<int> result;
  for (int degree = 1; degree <= max_degree; degree *= 2) {
    result.push_back(degree);
  }
  return result;
}

double estimate_llama2_strategy_memory_mib(BenchmarkArgs const &args,
                                           int data_parallel_degree,
                                           int linear_parallel_degree,
                                           int attention_parallel_degree) {
  int total_gpus = total_num_gpus(args);
  int local_batch = args.batch_size / data_parallel_degree;
  double layer_scale = static_cast<double>(args.decoder_layers) / 32.0;
  double seq_scale = static_cast<double>(args.sequence_length) / 1024.0;

  // Calibrated against the observed 32-layer FP32 training benchmark on this
  // cluster: pure DP16 replicates the full model and OOMs around 100+ GiB/GPU,
  // while DP4 with rotated layer placement lands in the 20-30 GiB range.
  constexpr double full_model_replica_mib = 88000.0;
  constexpr double linear_weight_fraction = 0.85;
  constexpr double attention_weight_fraction = 0.15;
  constexpr double mha_reserve_mib_per_task_batch1_seq1024 = 385.0;
  constexpr double mha_weight_mib_per_task = 256.0;
  constexpr double mha_runtime_state_safety_factor = 1.15;

  double widest_task_group =
      static_cast<double>(data_parallel_degree *
                          std::max(linear_parallel_degree,
                                   attention_parallel_degree));
  double placement_fraction =
      std::min(1.0, widest_task_group / static_cast<double>(total_gpus));
  double weight_split_fraction =
      linear_weight_fraction / static_cast<double>(linear_parallel_degree) +
      attention_weight_fraction /
          static_cast<double>(attention_parallel_degree);
  double model_mib =
      full_model_replica_mib * layer_scale * placement_fraction *
      weight_split_fraction;

  int attention_tasks_per_device =
      (args.decoder_layers * std::max(1, attention_parallel_degree) +
       args.num_gpus - 1) /
      args.num_gpus;
  double mha_runtime_state_mib =
      static_cast<double>(attention_tasks_per_device) *
      (mha_weight_mib_per_task +
       mha_reserve_mib_per_task_batch1_seq1024 *
           static_cast<double>(local_batch) * seq_scale * seq_scale) *
      mha_runtime_state_safety_factor;

  double activation_mib =
      1800.0 * layer_scale * seq_scale * static_cast<double>(local_batch) *
      placement_fraction /
      static_cast<double>(std::max(1, attention_parallel_degree));

  return model_mib + mha_runtime_state_mib + activation_mib +
         static_cast<double>(args.workspace_mb);
}

double estimate_llama2_strategy_cost(BenchmarkArgs const &args,
                                     int data_parallel_degree,
                                     int linear_parallel_degree,
                                     int attention_parallel_degree) {
  int searched_layers = args.search_decoder_layers > 0
                            ? args.search_decoder_layers
                            : args.decoder_layers;
  double local_batch =
      static_cast<double>(args.batch_size) /
      static_cast<double>(data_parallel_degree);
  double seq = static_cast<double>(args.sequence_length);
  double hidden = 4096.0;

  double linear_work =
      static_cast<double>(searched_layers) * local_batch * seq * hidden *
      hidden / static_cast<double>(linear_parallel_degree);
  double attention_work =
      static_cast<double>(searched_layers) * local_batch * seq * seq * hidden /
      static_cast<double>(attention_parallel_degree);
  double collective_penalty =
      static_cast<double>(searched_layers) *
      (1.0e-7 *
           std::log2(static_cast<double>(std::max(1, data_parallel_degree))) +
       1.0e-7 *
           std::log2(static_cast<double>(std::max(1, linear_parallel_degree))) +
       1.0e-7 *
           std::log2(static_cast<double>(std::max(1,
                                                  attention_parallel_degree))));

  constexpr double relative_linear_throughput = 3.0e13;
  constexpr double relative_attention_throughput = 1.2e14;
  return linear_work / relative_linear_throughput +
         attention_work / relative_attention_throughput + collective_penalty;
}

std::string reject_llama2_soap_candidate_reason(
    BenchmarkArgs const &args,
    int data_parallel_degree,
    int linear_parallel_degree,
    int attention_parallel_degree) {
  int total_gpus = total_num_gpus(args);
  if (args.batch_size % data_parallel_degree != 0) {
    return "global batch is not divisible by sample degree";
  }
  if (data_parallel_degree * linear_parallel_degree > total_gpus) {
    return "sample_degree * linear_degree exceeds available GPUs";
  }
  if (data_parallel_degree * attention_parallel_degree > total_gpus) {
    return "sample_degree * attention_degree exceeds available GPUs";
  }
  if (4096 % linear_parallel_degree != 0) {
    return "hidden size is not divisible by linear degree";
  }
  if (32 % attention_parallel_degree != 0) {
    return "number of attention heads is not divisible by attention degree";
  }
  if (data_parallel_degree == total_gpus && linear_parallel_degree == 1 &&
      attention_parallel_degree == 1 && args.decoder_layers >= 16) {
    return "pure data parallelism replicates the full FP32 model on every GPU";
  }
  int local_batch = args.batch_size / data_parallel_degree;
  if (local_batch > 1 && args.sequence_length >= 1024 &&
      args.decoder_layers >= 16 && attention_parallel_degree < local_batch) {
    return "cuDNN MHA reserve memory needs attention parallelism for this "
           "local batch";
  }
  double estimated_memory_mib =
      estimate_llama2_strategy_memory_mib(args,
                                          data_parallel_degree,
                                          linear_parallel_degree,
                                          attention_parallel_degree);
  double memory_limit_mib = static_cast<double>(args.fbmem_mb) * 0.97;
  if (estimated_memory_mib > memory_limit_mib) {
    std::ostringstream oss;
    oss << "estimated per-GPU memory " << estimated_memory_mib
        << " MiB exceeds safety limit " << memory_limit_mib << " MiB";
    return oss.str();
  }
  return "";
}

Llama2SoapSearchResult
run_llama2_soap_strategy_search(TransformerConfig const &full_config,
                                BenchmarkArgs const &args) {
  int total_gpus = total_num_gpus(args);
  int data_parallel_degree = args.data_parallel_degree;
  int max_tensor_degree = std::max(1, total_gpus / data_parallel_degree);
  int max_linear_degree =
      args.linear_parallel_degree > 1
          ? std::min(args.linear_parallel_degree, max_tensor_degree)
          : max_tensor_degree;
  int max_attention_degree =
      args.attention_parallel_degree > 0
          ? std::min(args.attention_parallel_degree, max_tensor_degree)
          : max_tensor_degree;
  std::vector<int> linear_degrees =
      power_of_two_degrees_up_to(max_linear_degree);
  std::vector<int> attention_degrees =
      power_of_two_degrees_up_to(max_attention_degree);

  TransformerConfig searched_config = full_config;
  if (args.search_decoder_layers > 0) {
    searched_config.num_decoder_layers =
        positive_int{args.search_decoder_layers};
  }

  std::cerr << "[llama2-benchmark] running LLaMA SOAP candidate search"
            << ": sample_degree=" << data_parallel_degree
            << ", searched_decoder_layers="
            << searched_config.num_decoder_layers.int_from_positive_int()
            << ", linear_degree_candidates=[";
  for (size_t i = 0; i < linear_degrees.size(); i++) {
    std::cerr << (i == 0 ? "" : ",") << linear_degrees.at(i);
  }
  std::cerr << "], attention_degree_candidates=[";
  for (size_t i = 0; i < attention_degrees.size(); i++) {
    std::cerr << (i == 0 ? "" : ",") << attention_degrees.at(i);
  }
  std::cerr << "]\n";

  std::optional<Llama2SoapStrategy> best_strategy = std::nullopt;
  size_t considered_candidates = 0;
  size_t feasible_candidates = 0;

  for (int linear_degree : linear_degrees) {
    for (int attention_degree : attention_degrees) {
      considered_candidates++;
      std::string rejection_reason =
          reject_llama2_soap_candidate_reason(args,
                                              data_parallel_degree,
                                              linear_degree,
                                              attention_degree);
      if (!rejection_reason.empty()) {
        std::cerr << "[llama2-benchmark] SOAP candidate rejected"
                  << ": sample=" << data_parallel_degree
                  << ", linear=" << linear_degree
                  << ", attention=" << attention_degree
                  << ", reason=" << rejection_reason << "\n";
        continue;
      }

      ParallelComputationGraph candidate_pcg =
          create_parallelized_llama2_benchmark_pcg(searched_config,
                                                   linear_degree,
                                                   data_parallel_degree,
                                                   attention_degree);
      size_t linear_layers =
          count_linear_parallelized_layers_from_pcg(candidate_pcg);
      size_t attention_layers =
          count_attention_parallelized_layers_from_pcg(candidate_pcg);
      if (linear_degree > 1 && linear_layers == 0) {
        std::cerr << "[llama2-benchmark] SOAP candidate rejected"
                  << ": sample=" << data_parallel_degree
                  << ", linear=" << linear_degree
                  << ", attention=" << attention_degree
                  << ", reason=no Linear operators were parallelized\n";
        continue;
      }
      if (attention_degree > 1 && attention_layers == 0) {
        std::cerr << "[llama2-benchmark] SOAP candidate rejected"
                  << ": sample=" << data_parallel_degree
                  << ", linear=" << linear_degree
                  << ", attention=" << attention_degree
                  << ", reason=no MHA operators were parallelized\n";
        continue;
      }

      double estimated_memory_mib =
          estimate_llama2_strategy_memory_mib(args,
                                              data_parallel_degree,
                                              linear_degree,
                                              attention_degree);
      double estimated_cost =
          estimate_llama2_strategy_cost(args,
                                        data_parallel_degree,
                                        linear_degree,
                                        attention_degree);
      feasible_candidates++;
      std::cerr << "[llama2-benchmark] SOAP candidate feasible"
                << ": sample=" << data_parallel_degree
                << ", linear=" << linear_degree
                << ", attention=" << attention_degree
                << ", estimated_cost=" << estimated_cost
                << ", estimated_memory_mib=" << estimated_memory_mib
                << ", linear_parallelized_layers=" << linear_layers
                << ", attention_parallelized_layers=" << attention_layers
                << "\n";

      Llama2SoapStrategy strategy{
          /*data_parallel_degree=*/data_parallel_degree,
          /*linear_parallel_degree=*/linear_degree,
          /*attention_parallel_degree=*/attention_degree,
          /*estimated_cost=*/estimated_cost,
          /*estimated_memory_mib=*/estimated_memory_mib,
          /*linear_parallelized_layers=*/linear_layers,
          /*attention_parallelized_layers=*/attention_layers,
      };
      if (!best_strategy.has_value() ||
          strategy.estimated_cost < best_strategy->estimated_cost) {
        best_strategy = strategy;
      }
    }
  }

  if (!best_strategy.has_value()) {
    std::ostringstream oss;
    oss << "LLaMA SOAP candidate search found no feasible strategy"
        << " (considered=" << considered_candidates
        << ", feasible=" << feasible_candidates << ")";
    throw std::runtime_error(oss.str());
  }

  std::cerr << "[llama2-benchmark] SOAP strategy selected"
            << ": sample=" << best_strategy->data_parallel_degree
            << ", linear=" << best_strategy->linear_parallel_degree
            << ", attention=" << best_strategy->attention_parallel_degree
            << ", estimated_cost=" << best_strategy->estimated_cost
            << ", estimated_memory_mib=" << best_strategy->estimated_memory_mib
            << ", considered_candidates=" << considered_candidates
            << ", feasible_candidates=" << feasible_candidates << "\n";

  ParallelComputationGraph full_pcg = create_parallelized_llama2_benchmark_pcg(
      full_config,
      best_strategy->linear_parallel_degree,
      best_strategy->data_parallel_degree,
      best_strategy->attention_parallel_degree);
  best_strategy->linear_parallelized_layers =
      count_linear_parallelized_layers_from_pcg(full_pcg);
  best_strategy->attention_parallelized_layers =
      count_attention_parallelized_layers_from_pcg(full_pcg);

  return Llama2SoapSearchResult{
      /*pcg=*/full_pcg,
      /*strategy=*/best_strategy.value(),
      /*used_search=*/true,
  };
}

std::vector<MachineView> valid_machine_views_with_same_layout(
    MachineView const &machine_view,
    OperatorTaskSpace const &task_space,
    MachineComputeSpecification const &machine_compute_spec) {
  std::vector<MachineView> result;
  MachineComputeResourceSlice machine_slice =
      compute_slice_from_specification(machine_compute_spec);
  int num_nodes = machine_compute_spec.num_nodes.int_from_positive_int();
  int num_gpus_per_node =
      machine_compute_spec.num_gpus_per_node.int_from_positive_int();

  for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
    for (int device_idx = 0; device_idx < num_gpus_per_node; device_idx++) {
      MachineView candidate = MachineView{
          MachineSpaceCoordinate{nonnegative_int{node_idx},
                                 nonnegative_int{device_idx},
                                 machine_view.start.device_type},
          machine_view.dimensions,
      };
      if (is_valid_machine_view(candidate, task_space, machine_slice)) {
        result.push_back(candidate);
      }
    }
  }

  std::sort(result.begin(), result.end(), [](MachineView const &lhs,
                                             MachineView const &rhs) {
    return std::tie(lhs.start.node_idx, lhs.start.device_idx) <
           std::tie(rhs.start.node_idx, rhs.start.device_idx);
  });
  return result;
}

MachineView rotate_machine_view_preserving_layout(
    MachineView const &machine_view,
    OperatorTaskSpace const &task_space,
    MachineComputeSpecification const &machine_compute_spec,
    int device_offset) {
  if (device_offset == 0 || machine_view.start.device_type != DeviceType::GPU) {
    return machine_view;
  }

  std::vector<MachineView> candidates = valid_machine_views_with_same_layout(
      machine_view, task_space, machine_compute_spec);
  if (candidates.empty()) {
    return machine_view;
  }

  auto current_it = std::find(candidates.begin(), candidates.end(),
                              machine_view);
  size_t current_idx =
      current_it == candidates.end()
          ? 0
          : static_cast<size_t>(std::distance(candidates.begin(), current_it));
  size_t rotated_idx =
      (current_idx + static_cast<size_t>(device_offset)) % candidates.size();
  return candidates.at(rotated_idx);
}

MachineMapping expand_decoder_layer_reused_mapping(
    ParallelComputationGraph const &full_pcg,
    SearchResult const &searched_result,
    MachineComputeSpecification const &machine_compute_spec,
    int full_decoder_layers, int searched_decoder_layers) {
  std::vector<parallel_layer_guid_t> full_order = topological_ordering(full_pcg);
  std::vector<parallel_layer_guid_t> searched_order =
      topological_ordering(searched_result.pcg);

  if (searched_decoder_layers <= 0 ||
      searched_decoder_layers >= full_decoder_layers) {
    return searched_result.machine_mapping;
  }

  size_t full_num_layers = full_order.size();
  size_t searched_num_layers = searched_order.size();
  size_t extra_decoder_layers =
      static_cast<size_t>(full_decoder_layers - searched_decoder_layers);
  if (full_num_layers <= searched_num_layers ||
      (full_num_layers - searched_num_layers) % extra_decoder_layers != 0) {
    throw std::runtime_error(
        "cannot infer repeated decoder-layer PCG size for mapping reuse");
  }

  size_t pcg_layers_per_decoder_layer =
      (full_num_layers - searched_num_layers) / extra_decoder_layers;
  size_t prefix_layers = 1;
  size_t searched_decoder_layer_pcg_layers =
      pcg_layers_per_decoder_layer *
      static_cast<size_t>(searched_decoder_layers);
  if (searched_num_layers < prefix_layers + searched_decoder_layer_pcg_layers) {
    throw std::runtime_error("searched PCG is smaller than expected for "
                             "decoder-layer mapping reuse");
  }
  size_t suffix_layers =
      searched_num_layers - prefix_layers - searched_decoder_layer_pcg_layers;
  size_t full_expected_num_layers =
      prefix_layers +
      pcg_layers_per_decoder_layer * static_cast<size_t>(full_decoder_layers) +
      suffix_layers;
  if (full_num_layers != full_expected_num_layers) {
    throw std::runtime_error(
        "full PCG size does not match inferred repeated decoder-layer layout");
  }

  std::unordered_map<parallel_layer_guid_t, MachineView> expanded_views;

  auto copy_mapping = [&](size_t full_idx, size_t searched_idx,
                          int device_offset) {
    parallel_layer_guid_t full_layer = full_order.at(full_idx);
    parallel_layer_guid_t searched_layer = searched_order.at(searched_idx);
    assert_reusable_layer_attrs_match(full_pcg, full_layer, searched_result.pcg,
                                      searched_layer);
    MachineView searched_machine_view =
        searched_result.machine_mapping.machine_views.at(searched_layer);
    OperatorTaskSpace full_task_space = get_operator_task_space(full_pcg,
                                                                full_layer);
    expanded_views.emplace(
        full_layer, rotate_machine_view_preserving_layout(searched_machine_view,
                                                          full_task_space,
                                                          machine_compute_spec,
                                                          device_offset));
  };

  for (size_t i = 0; i < prefix_layers; i++) {
    copy_mapping(i, i, 0);
  }

  for (int full_layer_idx = 0; full_layer_idx < full_decoder_layers;
       full_layer_idx++) {
    int searched_layer_idx = full_layer_idx % searched_decoder_layers;
    int device_offset = full_layer_idx / searched_decoder_layers;
    for (size_t layer_offset = 0; layer_offset < pcg_layers_per_decoder_layer;
         layer_offset++) {
      size_t full_idx =
          prefix_layers +
          static_cast<size_t>(full_layer_idx) * pcg_layers_per_decoder_layer +
          layer_offset;
      size_t searched_idx = prefix_layers +
                            static_cast<size_t>(searched_layer_idx) *
                                pcg_layers_per_decoder_layer +
                            layer_offset;
      copy_mapping(full_idx, searched_idx, device_offset);
    }
  }

  for (size_t i = 0; i < suffix_layers; i++) {
    size_t full_idx = prefix_layers +
                      pcg_layers_per_decoder_layer *
                          static_cast<size_t>(full_decoder_layers) +
                      i;
    size_t searched_idx = prefix_layers + searched_decoder_layer_pcg_layers + i;
    copy_mapping(full_idx, searched_idx, 0);
  }

  if (expanded_views.size() != full_num_layers) {
    throw std::runtime_error(
        "expanded decoder-layer mapping does not cover every PCG layer");
  }

  std::cerr << "[llama2-benchmark] decoder-layer mapping reuse: searched_pcg_"
               "layers="
            << searched_num_layers << ", full_pcg_layers=" << full_num_layers
            << ", pcg_layers_per_decoder_layer=" << pcg_layers_per_decoder_layer
            << ", suffix_layers=" << suffix_layers
            << ", device_rotation_period="
            << get_num_gpus(machine_compute_spec).int_from_positive_int()
            << "\n";

  return MachineMapping{/*machine_views=*/expanded_views};
}

std::string make_mapping_reuse_signature(ParallelComputationGraph const &pcg,
                                         parallel_layer_guid_t layer) {
  std::ostringstream oss;
  oss << pcg_op_attrs_get_op_type(pcg_get_op_attrs(pcg, layer)) << "|"
      << get_operator_task_space(pcg, layer);
  return oss.str();
}

MachineMapping expand_decoder_layer_reused_mapping_by_signature(
    ParallelComputationGraph const &full_pcg,
    SearchResult const &searched_result,
    MachineComputeSpecification const &machine_compute_spec,
    int full_decoder_layers, int searched_decoder_layers) {
  std::unordered_map<std::string, std::vector<MachineView>>
      searched_views_by_signature;
  for (parallel_layer_guid_t searched_layer :
       topological_ordering(searched_result.pcg)) {
    std::string signature =
        make_mapping_reuse_signature(searched_result.pcg, searched_layer);
    searched_views_by_signature[signature].push_back(
        searched_result.machine_mapping.machine_views.at(searched_layer));
  }

  std::unordered_map<std::string, size_t> next_candidate_by_signature;
  std::unordered_map<parallel_layer_guid_t, MachineView> expanded_views;
  for (parallel_layer_guid_t full_layer : topological_ordering(full_pcg)) {
    std::string signature = make_mapping_reuse_signature(full_pcg, full_layer);
    auto candidates_it = searched_views_by_signature.find(signature);
    if (candidates_it == searched_views_by_signature.end() ||
        candidates_it->second.empty()) {
      std::ostringstream oss;
      ParallelLayerAttrs full_attrs =
          get_parallel_layer_attrs(full_pcg, full_layer);
      oss << "cannot reuse decoder-layer mapping because no searched layer "
             "matches full layer signature: full_layer="
          << full_layer << ", full_name="
          << full_attrs.name.value_or("<unnamed>")
          << ", signature=" << signature;
      throw std::runtime_error(oss.str());
    }

    std::vector<MachineView> const &candidates = candidates_it->second;
    size_t occurrence = next_candidate_by_signature[signature]++;
    MachineView reused_view = candidates.at(occurrence % candidates.size());
    int device_offset = static_cast<int>(occurrence / candidates.size());
    OperatorTaskSpace full_task_space = get_operator_task_space(full_pcg,
                                                                full_layer);
    expanded_views.emplace(
        full_layer,
        rotate_machine_view_preserving_layout(reused_view,
                                              full_task_space,
                                              machine_compute_spec,
                                              device_offset));
  }

  std::cerr << "[llama2-benchmark] decoder-layer mapping reuse by signature: "
            << "searched_decoder_layers=" << searched_decoder_layers
            << ", full_decoder_layers=" << full_decoder_layers
            << ", searched_signatures="
            << searched_views_by_signature.size()
            << ", full_pcg_layers=" << expanded_views.size() << "\n";

  return MachineMapping{/*machine_views=*/expanded_views};
}

SearchResult run_auto_parallel_mapping_without_reuse(
    ParallelComputationGraph const &pcg,
    MachineComputeSpecification const &machine_compute_spec,
    BenchmarkArgs const &args) {
  MachineInterconnectSpecification interconnect_specification =
      make_machine_interconnect_specification(args);
  RuntimeOnlyCostEstimator cost_estimator =
      make_heuristic_runtime_only_cost_estimator(interconnect_specification);

  if (args.mapping_mode == MappingMode::UNITY) {
    std::cerr << "[llama2-benchmark] running unity auto-parallel search"
              << ": budget=" << args.search_budget
              << ", alpha=" << args.search_alpha
              << ", max_num_ops=" << args.search_max_num_ops << "\n";

    ParallelComputationGraph search_pcg = pcg;
    UnitySearchConfig search_config =
        UnitySearchConfig{/*alpha=*/args.search_alpha,
                          /*budget=*/args.search_budget,
                          /*max_num_ops=*/args.search_max_num_ops};
    return graph_optimize(search_pcg, cost_estimator, machine_compute_spec,
                          search_config);
  }

  if (args.mapping_mode == MappingMode::MCMC) {
    float effective_substitution_frequency = args.mcmc_substitution_frequency;
    if (args.data_parallel_degree > 1 &&
        effective_substitution_frequency > 0.0f) {
      std::cerr << "[llama2-benchmark] data_parallel_degree="
                << args.data_parallel_degree
                << " uses sample-parallel execution; disabling MCMC graph "
                   "substitutions for this run because mixed sample "
                   "parallelism + tensor-parallel substitutions are not "
                   "supported by the current copy/shard expansion path\n";
      effective_substitution_frequency = 0.0f;
    }

    std::cerr << "[llama2-benchmark] running mcmc auto-parallel search"
              << ": budget=" << args.search_budget
              << ", temperature=" << args.mcmc_temperature
              << ", substitution_frequency=" << effective_substitution_frequency
              << "\n";

    MachineSpecification full_machine_spec =
        MachineSpecification{/*compute_specification=*/machine_compute_spec,
                             /*interconnect_specification=*/
                             interconnect_specification};
    MCMCOverMappedPCGConfig search_config =
        MCMCOverMappedPCGConfig{/*temperature=*/args.mcmc_temperature,
                                /*num_iterations=*/
                                nonnegative_int{args.search_budget},
                                /*substitution_frequency=*/
                                effective_substitution_frequency,
                                /*device_type=*/DeviceType::GPU};
    return mcmc_over_mapped_pcg(pcg, cost_estimator, full_machine_spec,
                                search_config);
  }

  return create_balanced_mapping(pcg, machine_compute_spec);
}

SearchResult create_decoder_layer_reused_mcmc_mapping(
    ParallelComputationGraph const &full_pcg,
    MachineComputeSpecification const &machine_compute_spec,
    TransformerConfig const &full_config, BenchmarkArgs const &args) {
  if (args.search_decoder_layers <= 0 ||
      args.search_decoder_layers >= args.decoder_layers) {
    return run_auto_parallel_mapping_without_reuse(full_pcg,
                                                   machine_compute_spec, args);
  }

  std::cerr << "[llama2-benchmark] searching first "
            << args.search_decoder_layers
            << " decoder layer(s), then reusing the mapping across "
            << args.decoder_layers << " decoder layers\n";

  TransformerConfig searched_config = full_config;
  searched_config.num_decoder_layers = positive_int{args.search_decoder_layers};
  ParallelComputationGraph searched_pcg =
      (args.data_parallel_degree > 1)
          ? create_parallelized_llama2_benchmark_pcg(
                searched_config,
                args.linear_parallel_degree,
                args.data_parallel_degree)
          : pcg_from_computation_graph(
                get_decoder_only_transformer_computation_graph(
                    searched_config));
  ParallelComputationGraph reuse_full_pcg = full_pcg;

  BenchmarkArgs searched_args = args;
  searched_args.decoder_layers = args.search_decoder_layers;
  searched_args.search_decoder_layers = 0;
  size_t native_mcmc_layers_before = get_parallel_layers(searched_pcg).size();
  std::cerr << "[llama2-benchmark] running native FlexFlow MCMC on the "
               "searched decoder-layer window with requested "
               "substitution_frequency="
            << searched_args.mcmc_substitution_frequency
            << (args.data_parallel_degree > 1
                    ? " (data-parallel mode uses mapping-only MCMC)"
                    : "")
            << "\n";
  SearchResult searched_result = run_auto_parallel_mapping_without_reuse(
      searched_pcg, machine_compute_spec, searched_args);
  size_t native_mcmc_layers_after =
      get_parallel_layers(searched_result.pcg).size();
  std::cerr << "[llama2-benchmark] native FlexFlow MCMC searched-window "
               "result: layers_before="
            << native_mcmc_layers_before
            << ", layers_after=" << native_mcmc_layers_after
            << ", graph_changed="
            << (native_mcmc_layers_before != native_mcmc_layers_after ? "yes"
                                                                      : "no")
            << ", max_collective_degree="
            << get_max_parallel_collective_degree_from_pcg(searched_result.pcg)
            << ", max_linear_combine_degree="
            << get_max_combine_parallel_degree_from_pcg(searched_result.pcg)
            << ", max_attention_reduce_degree="
            << get_max_reduction_parallel_degree_from_pcg(searched_result.pcg)
            << "\n";

  int native_linear_degree =
      get_max_combine_parallel_degree_from_pcg(searched_result.pcg);
  int native_attention_degree =
      get_max_reduction_parallel_degree_from_pcg(searched_result.pcg);
  if (native_linear_degree > 1 || native_attention_degree > 1) {
    std::cerr << "[llama2-benchmark] native MCMC produced FlexFlow SOAP "
                 "substitutions; replaying the searched structure across the "
                 "full repeated graph: linear_degree="
              << native_linear_degree
              << ", attention_degree=" << native_attention_degree << "\n";

    LinearParallelizationResult full_linear_parallelized{
        /*pcg=*/reuse_full_pcg,
        /*stats=*/LinearParallelizationStats{}};
    if (native_linear_degree > 1) {
      full_linear_parallelized =
          apply_linear_parallelization(reuse_full_pcg, native_linear_degree);
      reuse_full_pcg = full_linear_parallelized.pcg;
    }

    AttentionParallelizationResult full_attention_parallelized{
        /*pcg=*/reuse_full_pcg,
        /*stats=*/AttentionParallelizationStats{}};
    if (native_attention_degree > 1) {
      full_attention_parallelized =
          apply_attention_parallelization(reuse_full_pcg,
                                          native_attention_degree);
      reuse_full_pcg = full_attention_parallelized.pcg;
    }

    if (native_linear_degree > 1 &&
        full_linear_parallelized.stats.transformed_linear_layers == 0) {
      throw std::runtime_error(
          "native MCMC produced Linear-parallel searched graph changes, but "
          "the same Linear structure could not be replayed on the full graph");
    }
    if (native_attention_degree > 1 &&
        full_attention_parallelized.stats.transformed_attention_layers == 0) {
      throw std::runtime_error(
          "native MCMC produced Attention-parallel searched graph changes, but "
          "the same Attention structure could not be replayed on the full "
          "graph");
    }

    std::cerr << "[llama2-benchmark] FlexFlow SOAP replay result: "
              << "full_layers_after=" << get_parallel_layers(reuse_full_pcg).size()
              << ", full_linear_substitutions="
              << full_linear_parallelized.stats.transformed_linear_layers
              << ", full_attention_substitutions="
              << full_attention_parallelized.stats.transformed_attention_layers
              << ", full_linear_parallelized_layers="
              << count_linear_parallelized_layers_from_pcg(reuse_full_pcg)
              << ", full_attention_parallelized_layers="
              << count_attention_parallelized_layers_from_pcg(reuse_full_pcg)
              << "\n";

    std::cerr << "[llama2-benchmark] reusing the native SOAP searched-window "
                 "mapping directly; skipping the extra mapping-only MCMC "
                 "remap\n";
  } else if (args.data_parallel_degree == 1 &&
             args.mcmc_substitution_frequency > 0.0f && args.num_gpus > 1) {
    int fallback_linear_degree = args.num_gpus;
    std::cerr << "[llama2-benchmark] native MCMC did not keep a "
                 "Linear-parallel graph change; applying the FlexFlow native "
                 "Linear substitution to the searched window and replaying it "
                 "across the full repeated graph: degree="
              << fallback_linear_degree << "\n";
    LinearParallelizationResult searched_parallelized =
        apply_linear_parallelization(searched_pcg, fallback_linear_degree);
    reuse_full_pcg =
        create_parallelized_llama2_benchmark_pcg(full_config,
                                                 fallback_linear_degree,
                                                 args.data_parallel_degree);
    if (searched_parallelized.stats.transformed_linear_layers == 0 ||
        count_linear_parallelized_layers_from_pcg(reuse_full_pcg) == 0) {
      log_linear_parallelization_candidates(searched_pcg,
                                            fallback_linear_degree);
      throw std::runtime_error(
          "FlexFlow Linear substitution did not match the searched/full "
          "Llama2 graph; see candidates above");
    }

    std::cerr << "[llama2-benchmark] FlexFlow Linear substitution fallback "
                 "result: searched_layers_before="
              << searched_parallelized.stats.pcg_layers_before
              << ", searched_layers_after="
              << searched_parallelized.stats.pcg_layers_after
              << ", searched_linear_substitutions="
              << searched_parallelized.stats.transformed_linear_layers
              << ", full_layers_after="
              << get_parallel_layers(reuse_full_pcg).size()
              << ", full_linear_parallelized_layers="
              << count_linear_parallelized_layers_from_pcg(reuse_full_pcg)
              << "\n";

    searched_result =
        create_balanced_mapping(searched_parallelized.pcg, machine_compute_spec);
    std::cerr << "[llama2-benchmark] reusing the FlexFlow-substituted searched "
                 "window with a fresh balanced mapping; skipping the extra "
                 "mapping-only MCMC remap\n";
  }

  return SearchResult{
      /*pcg=*/reuse_full_pcg,
      /*machine_mapping=*/
      expand_decoder_layer_reused_mapping_by_signature(
          reuse_full_pcg, searched_result, machine_compute_spec,
          args.decoder_layers, args.search_decoder_layers),
  };
}

SearchResult create_auto_parallel_mapping(
    ParallelComputationGraph const &pcg,
    MachineComputeSpecification const &machine_compute_spec,
    TransformerConfig const &config, BenchmarkArgs const &args) {
  if (args.search_decoder_layers > 0 &&
      args.mapping_mode == MappingMode::MCMC) {
    return create_decoder_layer_reused_mcmc_mapping(pcg, machine_compute_spec,
                                                    config, args);
  }

  return run_auto_parallel_mapping_without_reuse(pcg, machine_compute_spec,
                                                 args);
}

void run_synthetic_train_step(PCGInstance &pcg_instance,
                              DistributedFfHandle const &device_handle) {
  wait_for_events(perform_all_passes_for_pcg_instance(
      /*instance=*/pcg_instance,
      /*profiling_settings=*/ProfilingSettings{0, 1},
      /*device_handle=*/device_handle));
}

} // namespace

int main(int argc, char **argv) {
  std::string prog_name = argv[0];

  BenchmarkArgs args;
  try {
    args = parse_args(argc, argv);
  } catch (std::exception const &e) {
    print_usage(prog_name);
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  std::vector<char *> realm_args = make_realm_args(prog_name, args);
  int realm_argc = realm_args.size();
  char **realm_argv = realm_args.data();
  RealmManager manager(&realm_argc, &realm_argv);

  ControllerTaskResult result = manager.start_controller([&](RealmContext
                                                                 &ctx) {
    try {
      int realm_gpus = count_realm_processors(Realm::Processor::TOC_PROC);
      int realm_cpus = count_realm_processors(Realm::Processor::LOC_PROC);
      std::cerr << "[llama2-benchmark] Realm processors: gpus=" << realm_gpus
                << ", cpus=" << realm_cpus << "\n";
      if (realm_gpus < total_num_gpus(args)) {
        throw std::runtime_error(
            "requested " + std::to_string(total_num_gpus(args)) +
            " GPU(s), but Realm only created " + std::to_string(realm_gpus) +
            ". The preceding \"reservation ('GPU proc ...') cannot be "
            "satisfied\" message usually means --fbmem-mb is too high for the "
            "available GPU memory, or fewer GPUs were made visible to the "
            "job.");
      }

      std::cerr << "[llama2-benchmark] building computation graph\n";
      TransformerConfig config = get_llama2_7b_like_config();
      config.batch_size = positive_int{args.batch_size};
      config.sequence_length = positive_int{args.sequence_length};
      config.num_decoder_layers = positive_int{args.decoder_layers};
      std::cerr << "[llama2-benchmark] config: batch_size="
                << config.batch_size.int_from_positive_int()
                << ", seq_len=" << config.sequence_length.int_from_positive_int()
                << ", decoder_layers="
                << config.num_decoder_layers.int_from_positive_int()
                << ", nodes=" << args.num_nodes
                << ", gpus_per_node=" << args.num_gpus
                << ", total_gpus=" << total_num_gpus(args)
                << ", data_parallel_degree=" << args.data_parallel_degree
                << "\n";
      ParallelComputationGraph pcg =
          (args.data_parallel_degree > 1)
              ? create_parallelized_llama2_benchmark_pcg(
                    config,
                    args.linear_parallel_degree,
                    args.data_parallel_degree)
              : pcg_from_computation_graph(
                    get_decoder_only_transformer_computation_graph(config));
      if (args.data_parallel_degree > 1) {
        SampleParallelizationStats sample_stats =
            get_sample_parallelization_stats_from_pcg(pcg);
        std::cerr << "[llama2-benchmark] data-parallel PCG result: "
                  << "data_parallel_degree=" << args.data_parallel_degree
                  << ", sample_parallelized_layers="
                  << sample_stats.sample_parallelized_layers
                  << ", max_sample_parallel_degree="
                  << sample_stats.max_sample_parallel_degree
                  << ", pcg_layers=" << get_parallel_layers(pcg).size()
                  << "\n";
      }
      LinearParallelizationStats linear_parallelization_stats;
      if (args.linear_parallel_degree > 1 && args.data_parallel_degree == 1) {
        std::cerr << "[llama2-benchmark] applying focused Linear "
                     "parallelization: degree="
                  << args.linear_parallel_degree << "\n";
        LinearParallelizationResult parallelized =
            apply_linear_parallelization(pcg, args.linear_parallel_degree);
        pcg = parallelized.pcg;
        linear_parallelization_stats = parallelized.stats;
        std::cerr << "[llama2-benchmark] Linear parallelization result: "
                  << "original_linear_layers="
                  << linear_parallelization_stats.original_linear_layers
                  << ", transformed_linear_layers="
                  << linear_parallelization_stats.transformed_linear_layers
                  << ", pcg_layers_before="
                  << linear_parallelization_stats.pcg_layers_before
                  << ", pcg_layers_after="
                  << linear_parallelization_stats.pcg_layers_after << "\n";
        if (linear_parallelization_stats.transformed_linear_layers == 0) {
          log_linear_parallelization_candidates(pcg,
                                                args.linear_parallel_degree);
          throw std::runtime_error(
              "--linear-parallel-degree did not match any Linear operator; "
              "see the Linear parallelization candidates above for slots, "
              "output rank, and divisibility details");
        }
      }
      BenchmarkArgs execution_args = args;
      bool used_llama2_soap_strategy_search = false;
      Llama2SoapStrategy selected_soap_strategy;
      if (args.mapping_mode == MappingMode::MCMC &&
          args.data_parallel_degree > 1 && args.search_budget > 0) {
        Llama2SoapSearchResult soap_search_result =
            run_llama2_soap_strategy_search(config, args);
        pcg = soap_search_result.pcg;
        selected_soap_strategy = soap_search_result.strategy;
        used_llama2_soap_strategy_search = soap_search_result.used_search;
        execution_args.data_parallel_degree =
            selected_soap_strategy.data_parallel_degree;
        execution_args.linear_parallel_degree =
            selected_soap_strategy.linear_parallel_degree;
        linear_parallelization_stats.transformed_linear_layers =
            selected_soap_strategy.linear_parallelized_layers;
        linear_parallelization_stats.pcg_layers_after =
            get_parallel_layers(pcg).size();
        std::cerr << "[llama2-benchmark] LLaMA SOAP searched PCG result: "
                  << "sample_degree="
                  << selected_soap_strategy.data_parallel_degree
                  << ", linear_degree="
                  << selected_soap_strategy.linear_parallel_degree
                  << ", attention_degree="
                  << selected_soap_strategy.attention_parallel_degree
                  << ", linear_parallelized_layers="
                  << selected_soap_strategy.linear_parallelized_layers
                  << ", attention_parallelized_layers="
                  << selected_soap_strategy.attention_parallelized_layers
                  << ", pcg_layers=" << get_parallel_layers(pcg).size()
                  << "\n";
      }

      MachineComputeSpecification machine_compute_spec =
          MachineComputeSpecification{
              /*num_nodes=*/positive_int{args.num_nodes},
              /*num_cpus_per_node=*/positive_int{args.num_cpus},
              /*num_gpus_per_node=*/positive_int{args.num_gpus}};

      std::cerr << "[llama2-benchmark] mapping mode: "
                << mapping_mode_name(args.mapping_mode) << "\n";
      bool selected_soap_strategy_uses_native_substitutions =
          used_llama2_soap_strategy_search &&
          (selected_soap_strategy.linear_parallel_degree > 1 ||
           selected_soap_strategy.attention_parallel_degree > 1);
      bool use_flattened_dp_mapping =
          requires_flattened_data_parallel_mapping(execution_args) &&
          !selected_soap_strategy_uses_native_substitutions;
      bool use_direct_mapped_pcg =
          use_flattened_dp_mapping ||
          selected_soap_strategy_uses_native_substitutions;
      ParallelComputationGraph executable_pcg = pcg;
      size_t linear_parallelized_layers = 0;
      size_t attention_parallelized_layers = 0;
      SampleParallelizationStats sample_parallelization_stats;

      MappedParallelComputationGraph mpcg = [&]() {
        if (selected_soap_strategy_uses_native_substitutions) {
          std::cerr << "[llama2-benchmark] materializing SOAP-searched PCG "
                       "with direct FlexFlow mapped task groups; this keeps "
                       "the searched sample/linear/attention SOAP graph and "
                       "avoids large-graph MachineMapping construction.\n";
          executable_pcg = pcg;
          linear_parallelized_layers =
              std::max(linear_parallelization_stats.transformed_linear_layers,
                       count_linear_parallelized_layers_from_pcg(pcg));
          attention_parallelized_layers =
              count_attention_parallelized_layers_from_pcg(pcg);
          sample_parallelization_stats =
              get_sample_parallelization_stats_from_pcg(pcg);
          std::cerr << "[llama2-benchmark] materializing direct mapped PCG\n";
          MappedParallelComputationGraph direct_mpcg =
              create_flattened_data_parallel_mpcg(pcg, execution_args);
          print_mapped_pcg_summary(pcg, direct_mpcg);
          return direct_mpcg;
        }

        if (use_flattened_dp_mapping) {
          std::cerr << "[llama2-benchmark] using flattened data-parallel "
                       "mapping: data_parallel_degree="
                    << execution_args.data_parallel_degree
                    << ", nodes=" << execution_args.num_nodes
                    << ", gpus_per_node=" << execution_args.num_gpus << ".\n";
          if (execution_args.data_parallel_degree >
              max_single_machine_view_data_parallel_degree(execution_args)) {
            std::cerr << "[llama2-benchmark] MachineView cannot express "
                         "sample-parallel degree "
                      << execution_args.data_parallel_degree << " on a "
                      << execution_args.num_nodes << "x"
                      << execution_args.num_gpus
                      << " machine as a single dimension, so the benchmark "
                         "materializes the sample shard space directly onto "
                         "the flattened node/GPU list.\n";
          } else {
            std::cerr << "[llama2-benchmark] direct data-parallel mapping "
                         "uses the same FlexFlow sample-parallel tensor "
                         "semantics but avoids expensive MachineView search "
                         "for the large repeated LLaMA graph. Layer mappings "
                         "are rotated over the flattened node/GPU list.\n";
          }
          if (args.mapping_mode == MappingMode::MCMC &&
              used_llama2_soap_strategy_search) {
            std::cerr << "[llama2-benchmark] flattened data-parallel mapping "
                         "uses the PCG selected by the LLaMA SOAP candidate "
                         "search; only the MachineView placement MCMC is "
                         "skipped for this large repeated graph.\n";
          } else if (args.mapping_mode == MappingMode::MCMC) {
            std::cerr << "[llama2-benchmark] flattened data-parallel mapping "
                         "skips MachineView-based MCMC search for this run; "
                         "the PCG still uses FlexFlow sample-parallel tensor "
                         "semantics and replicated weights.\n";
          }
          executable_pcg = pcg;
          linear_parallelized_layers =
              std::max(linear_parallelization_stats.transformed_linear_layers,
                       count_linear_parallelized_layers_from_pcg(pcg));
          attention_parallelized_layers =
              count_attention_parallelized_layers_from_pcg(pcg);
          sample_parallelization_stats =
              get_sample_parallelization_stats_from_pcg(pcg);
          std::cerr << "[llama2-benchmark] materializing flattened "
                       "data-parallel mapped PCG\n";
          MappedParallelComputationGraph direct_mpcg =
              create_flattened_data_parallel_mpcg(pcg, execution_args);
          print_mapped_pcg_summary(pcg, direct_mpcg);
          return direct_mpcg;
        }

        SearchResult mapping_result =
            create_auto_parallel_mapping(pcg, machine_compute_spec, config,
                                         execution_args);
        executable_pcg = mapping_result.pcg;
        linear_parallelized_layers =
            std::max(linear_parallelization_stats.transformed_linear_layers,
                     count_linear_parallelized_layers_from_pcg(
                         mapping_result.pcg));
        attention_parallelized_layers =
            count_attention_parallelized_layers_from_pcg(mapping_result.pcg);
        sample_parallelization_stats =
            get_sample_parallelization_stats_from_pcg(mapping_result.pcg);
        print_mapping_summary(mapping_result.pcg,
                              mapping_result.machine_mapping);
        std::cerr << "[llama2-benchmark] materializing mapped PCG\n";
        return mapped_pcg_from_pcg_and_mapping(mapping_result.pcg,
                                               mapping_result.machine_mapping);
      }();

      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                           /*momentum=*/0.0,
                                           /*nesterov=*/false,
                                           /*weight_decay=*/0.001}};

      std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
          input_tensors;

      std::cerr << "[llama2-benchmark] creating distributed FF handle\n";
      DistributedFfHandle device_handle = create_distributed_ff_handle(
          ctx,
          /*workSpaceSize=*/static_cast<size_t>(args.workspace_mb) * 1024 *
              1024,
          /*allowTensorOpMathConversion=*/true);

      std::cerr << "[llama2-benchmark] creating PCG instance\n";
      PCGInstance pcg_instance = create_pcg_instance(
          /*ctx=*/ctx,
          /*mpcg=*/mpcg,
          /*optimizer=*/optimizer_attrs,
          /*loss=*/std::nullopt,
          /*input_tensors=*/input_tensors,
          /*profiling_settings=*/
          ProfilingSettings{args.warmup_iters, args.measure_iters},
          /*device_handle=*/device_handle);

      std::cerr << "[llama2-benchmark] running warmup\n";
      for (int i = 0; i < args.warmup_iters; i++) {
        run_synthetic_train_step(pcg_instance, device_handle);
      }

      std::cerr << "[llama2-benchmark] running measured iterations\n";
      double batch_size =
          static_cast<double>(config.batch_size.int_from_positive_int());
      std::vector<double> step_seconds;
      step_seconds.reserve(args.measure_iters);

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < args.measure_iters; i++) {
        auto step_start = std::chrono::steady_clock::now();
        run_synthetic_train_step(pcg_instance, device_handle);
        auto step_stop = std::chrono::steady_clock::now();
        step_seconds.push_back(
            std::chrono::duration_cast<std::chrono::duration<double>>(
                step_stop - step_start)
                .count());
      }
      auto stop = std::chrono::steady_clock::now();

      double seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(stop -
                                                                    start)
              .count();
      size_t avg_window_steps =
          std::min<size_t>(static_cast<size_t>(5), step_seconds.size());
      double avg_window_seconds = 0.0;
      for (size_t i = step_seconds.size() - avg_window_steps;
           i < step_seconds.size(); i++) {
        avg_window_seconds += step_seconds.at(i);
      }
      double samples = batch_size * static_cast<double>(avg_window_steps);
      double samples_per_second = samples / avg_window_seconds;
      double step_ms = 1000.0 * avg_window_seconds / avg_window_steps;

      std::cout << "model=llama2_7b_like\n";
      std::cout << "global_batch_size="
                << config.batch_size.int_from_positive_int() << "\n";
      std::cout << "sequence_length="
                << config.sequence_length.int_from_positive_int() << "\n";
      std::cout << "hidden_size=" << config.num_features.int_from_positive_int()
                << "\n";
      std::cout << "num_layers="
                << config.num_decoder_layers.int_from_positive_int() << "\n";
      std::cout << "num_heads=" << config.num_heads.int_from_positive_int()
                << "\n";
      std::cout << "nodes=" << args.num_nodes << "\n";
      std::cout << "gpus_per_node=" << args.num_gpus << "\n";
      std::cout << "gpus=" << total_num_gpus(args) << "\n";
      std::cout << "mapping=" << mapping_mode_name(args.mapping_mode) << "\n";
      std::cout << "search_budget=" << args.search_budget << "\n";
      std::cout << "search_decoder_layers=" << args.search_decoder_layers
                << "\n";
      std::cout << "mcmc_substitution_frequency="
                << args.mcmc_substitution_frequency << "\n";
      std::cout << "effective_mcmc_substitution_frequency="
                << (execution_args.data_parallel_degree > 1 &&
                            !used_llama2_soap_strategy_search
                        ? 0.0f
                        : args.mcmc_substitution_frequency)
                << "\n";
      std::cout << "flattened_data_parallel_mapping="
                << (use_direct_mapped_pcg ? 1 : 0) << "\n";
      std::cout << "data_parallel_degree="
                << execution_args.data_parallel_degree
                << "\n";
      std::cout << "sample_parallelized_layers="
                << sample_parallelization_stats.sample_parallelized_layers
                << "\n";
      std::cout << "max_sample_parallel_degree="
                << sample_parallelization_stats.max_sample_parallel_degree
                << "\n";
      std::cout << "linear_parallel_degree="
                << execution_args.linear_parallel_degree
                << "\n";
      std::cout << "attention_parallel_degree="
                << (used_llama2_soap_strategy_search
                        ? selected_soap_strategy.attention_parallel_degree
                        : get_max_reduction_parallel_degree_from_pcg(
                              executable_pcg))
                << "\n";
      std::cout << "llama2_soap_strategy_search="
                << (used_llama2_soap_strategy_search ? 1 : 0) << "\n";
      if (used_llama2_soap_strategy_search) {
        std::cout << "llama2_soap_estimated_cost="
                  << selected_soap_strategy.estimated_cost << "\n";
        std::cout << "llama2_soap_estimated_memory_mib="
                  << selected_soap_strategy.estimated_memory_mib << "\n";
      }
      std::cout << "linear_parallelized_layers="
                << linear_parallelized_layers << "\n";
      std::cout << "attention_parallelized_layers="
                << attention_parallelized_layers << "\n";
      std::cout << "soap_parallelized_layers="
                << (linear_parallelized_layers + attention_parallelized_layers)
                << "\n";
      std::cout << "warmup_iters=" << args.warmup_iters << "\n";
      std::cout << "measure_iters=" << args.measure_iters << "\n";
      for (size_t i = 0; i < step_seconds.size(); i++) {
        double current_step_ms = 1000.0 * step_seconds.at(i);
        double current_samples_per_second = batch_size / step_seconds.at(i);
        std::cout << "step=" << (i + 1) << " step_ms=" << current_step_ms
                  << " samples_per_second=" << current_samples_per_second
                  << "\n";
      }
      std::cout << "avg_window_steps=" << avg_window_steps << "\n";
      std::cout << "avg_step_ms=" << step_ms << "\n";
      std::cout << "avg_samples_per_second=" << samples_per_second << "\n";
    } catch (std::exception const &e) {
      std::cerr << "[llama2-benchmark] failed: " << e.what() << "\n";
      throw;
    }
  });

  result.wait();
  return 0;
}
