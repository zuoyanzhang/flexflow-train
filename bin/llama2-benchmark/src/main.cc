#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_mutation_set.h"
#include "models/transformer/transformer.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/pcg_from_computation_graph.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/pcg_instance.h"
#include "realm-execution/realm_manager.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/positive_int/positive_int.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace FlexFlow;

namespace {

struct BenchmarkArgs {
  int num_cpus = 4;
  int num_gpus = 1;
  int fbmem_mb = 64000;
  int zcmem_mb = 4096;
  int workspace_mb = 1024;
  int warmup_iters = 1;
  int measure_iters = 5;
};

char *leak_string_contents(std::string_view str) {
  std::vector<char> *content = new std::vector<char>{str.begin(), str.end()};
  content->push_back(0);
  return content->data();
}

std::vector<char *> make_realm_args(std::string_view executable_name,
                                    BenchmarkArgs const &args) {
  std::vector<std::string> values = {
      std::string{executable_name},
      "-ll:cpu",
      std::to_string(args.num_cpus),
      "-ll:gpu",
      std::to_string(args.num_gpus),
      "-ll:fsize",
      std::to_string(args.fbmem_mb),
      "-ll:zsize",
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
            << " [--gpus N] [--cpus N] [--fbmem-mb MB] [--zcmem-mb MB]"
               " [--workspace-mb MB] [--warmup N] [--iters N]\n";
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

    if (arg == "--gpus") {
      result.num_gpus = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--cpus") {
      result.num_cpus = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--fbmem-mb") {
      result.fbmem_mb = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--zcmem-mb") {
      result.zcmem_mb = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--workspace-mb") {
      result.workspace_mb = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--warmup") {
      result.warmup_iters = parse_nonnegative_int_arg(arg, read_value());
    } else if (arg == "--iters") {
      result.measure_iters = parse_positive_int_arg(arg, read_value());
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return result;
}

template <typename EventMap>
void wait_for_events(EventMap const &events) {
  using Event = std::decay_t<decltype(events.begin()->second)>;

  std::vector<Event> event_values;
  event_values.reserve(events.size());
  for (auto const &kv : events) {
    event_values.push_back(kv.second);
  }
  Event::merge_events(event_values).wait();
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

  ControllerTaskResult result = manager.start_controller([&](RealmContext &ctx) {
    TransformerConfig config = get_llama2_7b_like_config();
    ComputationGraph cg = get_llama2_7b_like_computation_graph();
    ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

    MachineComputeSpecification machine_spec =
        MachineComputeSpecification{/*num_nodes=*/1_p,
                                    /*num_cpus_per_node=*/positive_int{
                                        args.num_cpus},
                                    /*num_gpus_per_node=*/positive_int{
                                        args.num_gpus}};

    std::optional<MachineMapping> maybe_mapping =
        get_random_mapping(pcg, machine_spec, DeviceType::GPU);
    if (!maybe_mapping.has_value()) {
      throw std::runtime_error("failed to create a random GPU mapping for "
                               "llama2_7b_like");
    }
    MappedParallelComputationGraph mpcg =
        mapped_pcg_from_pcg_and_mapping(pcg, maybe_mapping.value());

    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                         /*momentum=*/0.9,
                                         /*nesterov=*/false,
                                         /*weight_decay=*/0.001}};

    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

    DistributedFfHandle device_handle = create_distributed_ff_handle(
        ctx,
        /*workSpaceSize=*/static_cast<size_t>(args.workspace_mb) * 1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    PCGInstance pcg_instance = create_pcg_instance(
        /*ctx=*/ctx,
        /*mpcg=*/mpcg,
        /*optimizer=*/optimizer_attrs,
        /*loss=*/std::nullopt,
        /*input_tensors=*/input_tensors,
        /*profiling_settings=*/ProfilingSettings{args.warmup_iters,
                                                 args.measure_iters},
        /*device_handle=*/device_handle);

    for (int i = 0; i < args.warmup_iters; i++) {
      run_synthetic_train_step(pcg_instance, device_handle);
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < args.measure_iters; i++) {
      run_synthetic_train_step(pcg_instance, device_handle);
    }
    auto stop = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    double samples =
        static_cast<double>(config.batch_size.int_from_positive_int()) *
        static_cast<double>(args.measure_iters);
    double samples_per_second = samples / seconds;
    double step_ms = 1000.0 * seconds / args.measure_iters;

    std::cout << "model=llama2_7b_like\n";
    std::cout << "global_batch_size="
              << config.batch_size.int_from_positive_int() << "\n";
    std::cout << "sequence_length="
              << config.sequence_length.int_from_positive_int() << "\n";
    std::cout << "hidden_size="
              << config.num_features.int_from_positive_int() << "\n";
    std::cout << "num_layers="
              << config.num_decoder_layers.int_from_positive_int() << "\n";
    std::cout << "num_heads=" << config.num_heads.int_from_positive_int()
              << "\n";
    std::cout << "gpus=" << args.num_gpus << "\n";
    std::cout << "warmup_iters=" << args.warmup_iters << "\n";
    std::cout << "measure_iters=" << args.measure_iters << "\n";
    std::cout << "step_ms=" << step_ms << "\n";
    std::cout << "samples_per_second=" << samples_per_second << "\n";
  });

  result.wait();
  return 0;
}
