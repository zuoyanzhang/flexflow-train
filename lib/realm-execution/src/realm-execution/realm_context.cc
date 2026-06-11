#include "realm-execution/realm_context.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_handle_t.h"
#include "kernels/device.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "pcg/device_id.h"
#include "pcg/device_id_t.h"
#include "pcg/device_type.dtg.h"
#include "realm-execution/realm_allocator.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/one_to_many/one_to_many.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

namespace {

struct NoCurrentProcessorAllocator : public IAllocator {
  void *allocate(size_t) override {
    PANIC("Cannot allocate through a RealmContext with no current processor");
    return nullptr;
  }

  void deallocate(void *) override {
    PANIC("Cannot deallocate through a RealmContext with no current processor");
  }

  DeviceType get_allocation_device_type() const override {
    PANIC("A RealmContext with no current processor has no allocation device "
          "type");
    return DeviceType::CPU;
  }
};

Allocator get_allocator_for_processor(Realm::Processor processor) {
  if (!processor.exists()) {
    return Allocator::create<NoCurrentProcessorAllocator>();
  }

  return get_realm_allocator(processor,
                             RealmContext::get_nearest_memory(processor));
}

} // namespace

RealmContext::RealmContext(Realm::Processor processor)
    : processor(processor), allocator(get_allocator_for_processor(processor)) {}

RealmContext::~RealmContext() {
  if (!this->outstanding_events.empty()) {
    Realm::Event outstanding = this->merge_outstanding_events();
    outstanding.wait();
  }
}

static std::tuple<Realm::AddressSpace, Realm::Processor::Kind, nonnegative_int>
convert_machine_space_coordinate(MachineSpaceCoordinate const &device_coord) {
  Realm::AddressSpace as = int{device_coord.node_idx};
  Realm::Processor::Kind kind;
  switch (device_coord.device_type) {
  case DeviceType::CPU:
    kind = Realm::Processor::Kind::LOC_PROC;
    break;
  case DeviceType::GPU:
    kind = Realm::Processor::Kind::TOC_PROC;
    break;
  default:
    PANIC("Unhandled DeviceType", fmt::to_string(device_coord.device_type));
    break;
  }
  nonnegative_int proc_in_node = device_coord.device_idx;
  return std::tuple{as, kind, proc_in_node};
}

Realm::Processor RealmContext::map_device_coord_to_processor(
    MachineSpaceCoordinate const &device_coord) {
  this->discover_machine_topology();
  auto [as, kind, proc_in_node] =
      convert_machine_space_coordinate(device_coord);
  auto key = std::pair{as, kind};
  auto processors_for_kind = this->processors.find(key);
  if (processors_for_kind == this->processors.end()) {
    throw mk_runtime_error(fmt::format(
        "No Realm processors found for requested device coordinate {}. "
        "If the log contains \"reservation ('GPU proc ...') cannot be "
        "satisfied\", reduce --fbmem-mb or request fewer GPUs.",
        device_coord));
  }

  size_t idx = proc_in_node.size_t_from_nonnegative_int();
  if (idx >= processors_for_kind->second.size()) {
    throw mk_runtime_error(fmt::format(
        "Requested device coordinate {} but Realm only created {} processors "
        "for address_space={} kind={}. If the log contains \"reservation "
        "('GPU proc ...') cannot be satisfied\", reduce --fbmem-mb or request "
        "fewer GPUs.",
        device_coord, processors_for_kind->second.size(), as,
        static_cast<int>(kind)));
  }

  return processors_for_kind->second.at(idx);
}

Realm::Memory RealmContext::get_nearest_memory(Realm::Processor proc) {
  if (!proc.exists()) {
    throw mk_runtime_error("Cannot find nearest memory for a non-existent "
                           "Realm processor");
  }

  // FIMXE: this isn't going to do what you expect until
  // https://github.com/StanfordLegion/realm/pull/392 merges
  Realm::Machine::MemoryQuery mq(Realm::Machine::get_machine());
  mq.best_affinity_to(proc);
  if (mq.count() == 0) {
    throw mk_runtime_error(fmt::format(
        "Realm processor {} has no visible memory with affinity. This usually "
        "means Realm failed to reserve GPU framebuffer memory; try reducing "
        "--fbmem-mb.",
        proc.id));
  }
  return mq.first();
}

Realm::Processor RealmContext::get_current_processor() const {
  return this->processor;
}

Allocator &RealmContext::get_current_device_allocator() {
  return this->allocator;
}

device_id_t RealmContext::get_current_device_idx() const {
  Realm::Processor proc = this->get_current_processor();

  // FIXME: find a more efficient way to implement this than scanning the
  // machine every time
  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  pq.same_address_space_as(proc);
  pq.only_kind(proc.kind());
  nonnegative_int idx{0};
  for (Realm::Processor p : pq) {
    if (p == proc) {
      break;
    }
    idx++;
  }

  switch (proc.kind()) {
  case Realm::Processor::LOC_PROC:
    return make_device_id_t_from_idx(idx, DeviceType::CPU);
  case Realm::Processor::TOC_PROC:
    return make_device_id_t_from_idx(idx, DeviceType::GPU);
  default:
    PANIC("Unhandled Realm::ProcessorKind", fmt::to_string(int{proc.kind()}));
  }
}

void RealmContext::set_cuda_device_for_current_processor() const {
  device_id_t device_idx = this->get_current_device_idx();
  if (!device_idx.is_gpu()) {
    return;
  }

  int cuda_device = get_raw_id(device_idx).int_from_nonnegative_int();
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudaError_t status = cudaSetDevice(cuda_device);
#elif defined(FF_USE_HIP_ROCM)
  hipError_t status = hipSetDevice(cuda_device);
#else
#error "Unknown device"
#endif
  checkCUDA(status);
}

Realm::Event
RealmContext::spawn_task(Realm::Processor proc, task_id_t task_id,
                         void const *args, size_t arglen,
                         Realm::ProfilingRequestSet const &requests,
                         Realm::Event wait_on, int priority) {
  Realm::Event result = proc.spawn(get_realm_task_id_for_task_id(task_id), args,
                                   arglen, requests, wait_on, priority);
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event RealmContext::collective_spawn_task(
    Realm::Processor target_proc, task_id_t task_id, void const *args,
    size_t arglen, Realm::Event wait_on, int priority) {
  Realm::Event result = this->runtime.collective_spawn(
      target_proc, get_realm_task_id_for_task_id(task_id), args, arglen,
      wait_on, priority);
  this->outstanding_events.push_back(result);
  return result;
}

template <int N, typename T = int>
static Realm::Rect<N, T> rect_from_dims(TensorDims const &dims) {
  std::vector<int> values{dims.ff_ordered.begin(), dims.ff_ordered.end()};
  ASSERT(values.size() == N);
  return Realm::Rect<N, T>{Realm::Point<N, T>::ZEROES(),
                           Realm::Point<N, T>{values.data()} -
                               Realm::Point<N, T>::ONES()};
}

template <int N, typename T = int>
static Realm::IndexSpace<N, T> ispace_from_dims(TensorDims const &dims) {
  Realm::Rect<N, T> rect = rect_from_dims<N, T>(dims);
  return Realm::IndexSpace<N, T>{rect};
}

Realm::Event RealmContext::issue_copy(
    ParallelTensorShape const &src_shape, Realm::RegionInstance src_inst,
    ParallelTensorShape const &dst_shape, Realm::RegionInstance dst_inst,
    Realm::ProfilingRequestSet const &requests, Realm::Event wait_on,
    int priority) {
  TensorShape src_piece_shape = get_piece_shape(src_shape);
  TensorShape dst_piece_shape = get_piece_shape(dst_shape);
  ASSERT(src_piece_shape == dst_piece_shape); // For now, assume they match

  Realm::CopySrcDstField src_field;
  src_field.set_field(
      /*inst=*/src_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);
  Realm::CopySrcDstField dst_field;
  dst_field.set_field(
      /*inst=*/dst_inst,
      /*field_id=*/0,
      /*size=*/
      static_cast<size_t>(
          size_of_datatype(src_piece_shape.data_type).int_from_positive_int()),
      /*subfield_offset=*/0);

  Realm::Event result;
  switch (src_piece_shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
  case 1:
    result = ispace_from_dims<1>(src_piece_shape.dims)
                 .copy({src_field}, {dst_field}, requests, wait_on, priority);
    break;
#endif
#if REALM_MAX_DIM >= 2
  case 2:
    result = ispace_from_dims<2>(src_piece_shape.dims)
                 .copy({src_field}, {dst_field}, requests, wait_on, priority);
    break;
#endif
#if REALM_MAX_DIM >= 3
  case 3:
    result = ispace_from_dims<3>(src_piece_shape.dims)
                 .copy({src_field}, {dst_field}, requests, wait_on, priority);
    break;
#endif
#if REALM_MAX_DIM >= 4
  case 4:
    result = ispace_from_dims<4>(src_piece_shape.dims)
                 .copy({src_field}, {dst_field}, requests, wait_on, priority);
    break;
#endif
#if REALM_MAX_DIM >= 5
  case 5:
    result = ispace_from_dims<5>(src_piece_shape.dims)
                 .copy({src_field}, {dst_field}, requests, wait_on, priority);
    break;
#endif
  default:
    PANIC("TensorShape dims greater than REALM_MAX_DIM: {}",
          src_piece_shape.dims.ff_ordered.num_dims());
    break;
  }
  this->outstanding_events.push_back(result);
  return result;
}

std::pair<Realm::RegionInstance, Realm::Event>
RealmContext::create_instance(Realm::Memory memory, TensorShape const &shape,
                              Realm::ProfilingRequestSet const &prs,
                              Realm::Event wait_on) {
  if (!memory.exists()) {
    throw mk_runtime_error(fmt::format(
        "Cannot create Realm instance for shape {} because target memory does "
        "not exist",
        shape));
  }

  std::vector<size_t> field_sizes{static_cast<size_t>(
      size_of_datatype(shape.data_type).int_from_positive_int())};
  Realm::RegionInstance inst;
  Realm::Event ready;
  switch (shape.dims.ff_ordered.num_dims()) {
#if REALM_MAX_DIM >= 1
  case 1:
    ready = Realm::RegionInstance::create_instance(
        inst, memory, rect_from_dims<1>(shape.dims), field_sizes, 0 /*SOA*/,
        prs, wait_on);
    break;
#endif
#if REALM_MAX_DIM >= 2
  case 2:
    ready = Realm::RegionInstance::create_instance(
        inst, memory, rect_from_dims<2>(shape.dims), field_sizes, 0 /*SOA*/,
        prs, wait_on);
    break;
#endif
#if REALM_MAX_DIM >= 3
  case 3:
    ready = Realm::RegionInstance::create_instance(
        inst, memory, rect_from_dims<3>(shape.dims), field_sizes, 0 /*SOA*/,
        prs, wait_on);
    break;
#endif
#if REALM_MAX_DIM >= 4
  case 4:
    ready = Realm::RegionInstance::create_instance(
        inst, memory, rect_from_dims<4>(shape.dims), field_sizes, 0 /*SOA*/,
        prs, wait_on);
    break;
#endif
#if REALM_MAX_DIM >= 5
  case 5:
    ready = Realm::RegionInstance::create_instance(
        inst, memory, rect_from_dims<5>(shape.dims), field_sizes, 0 /*SOA*/,
        prs, wait_on);
    break;
#endif
  default:
    PANIC("TensorShape dims greater than REALM_MAX_DIM",
          fmt::to_string(shape.dims.ff_ordered.num_dims()));
    break;
  }
  if (!inst.exists()) {
    size_t bytes = get_size_in_bytes(shape)
                       .unwrap_num_bytes()
                       .size_t_from_nonnegative_int();
    throw mk_runtime_error(fmt::format(
        "Realm failed to create an instance in memory {} for shape {} "
        "({} MiB). This usually means GPU framebuffer memory is exhausted; "
        "try lowering --fbmem-mb, using a smaller benchmark shape, or using a "
        "memory-aware/tensor-parallel mapping.",
        memory.id, shape, static_cast<double>(bytes) / (1024.0 * 1024.0)));
  }
  this->outstanding_events.push_back(ready);
  return std::pair{inst, ready};
}

Realm::Event RealmContext::get_outstanding_events() {
  Realm::Event result = this->merge_outstanding_events();
  this->outstanding_events.push_back(result);
  return result;
}

Realm::Event RealmContext::merge_outstanding_events() {
  Realm::Event result = Realm::Event::merge_events(this->outstanding_events);
  this->outstanding_events.clear();
  return result;
}

void RealmContext::discover_machine_topology() {
  if (!this->processors.empty()) {
    return;
  }

  Realm::Machine::ProcessorQuery pq(Realm::Machine::get_machine());
  for (Realm::Processor proc : pq) {
    Realm::AddressSpace as = proc.address_space();
    Realm::Processor::Kind kind = proc.kind();
    this->processors[std::pair{as, kind}].push_back(proc);
  }
}

Realm::Runtime RealmContext::get_runtime() { return this->runtime; }

} // namespace FlexFlow
