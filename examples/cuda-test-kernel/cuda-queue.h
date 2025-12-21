#pragma once

#include "dmtmacros.h"

#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
#  include "platform/platform-context.h"
#  include "cuda-wrappers/cuda-wrappers-utils.h"
#  include <algorithm>
#  include <type_traits>

#  include <vector>
#  include <cstddef>
#  include <type_traits>

namespace dmt {
template <typename... Ts>
void populateSizesAndAlignments(std::vector<size_t>& sizes,
                                std::vector<size_t>& alignments) {
  // Reserve space for efficiency
  sizes.reserve(sizeof...(Ts));
  alignments.reserve(sizeof...(Ts));

  // Fold expression over the comma operator
  ((sizes.push_back(sizeof(Ts)), alignments.push_back(alignof(Ts))), ...);
}
}  // namespace dmt
#endif

#if defined(__CUDA_ARCH__)
// ------------------ Metafunctions ------------------

// -------------------- index helpers --------------------
// clang-format off
template <size_t... Is> struct IndexSequence {};
template <size_t N, size_t... Is> struct MakeIndexSequence : MakeIndexSequence<N-1, N-1, Is...> {};
template <size_t... Is> struct MakeIndexSequence<0, Is...> { using type = IndexSequence<Is...>; };
template <typename... Ts> using IndexSequenceFor = typename MakeIndexSequence<sizeof...(Ts)>::type;
template <size_t N> struct IC { static constexpr size_t value = N; };

// -------------------- type-at helper --------------------
template <size_t I, typename... Ts> struct TypeAt;
template <typename T, typename... Ts> struct TypeAt<0, T, Ts...> { using type = T; };
template <size_t I, typename T, typename... Ts>
struct TypeAt<I, T, Ts...> { using type = typename TypeAt<I-1, Ts...>::type; };

// -------------------- PtrHolder / PtrTuple (stores pointers) --------------------
template <size_t I, typename T> struct PtrHolder { T* ptr; };

template <typename IndexSeq, typename... Ts> struct PtrTupleImpl;
template <size_t... Is, typename... Ts>
struct PtrTupleImpl<IndexSequence<Is...>, Ts...> : PtrHolder<Is, Ts>... {
    __device__ PtrTupleImpl(Ts*... ps) : PtrHolder<Is, Ts>{ps}... {}
    template <size_t I>
    __device__ auto get() -> typename TypeAt<I, Ts...>::type* {
        return this->template PtrHolder<I, typename TypeAt<I, Ts...>::type>::ptr;
    }
};
template <typename... Ts> using PtrTuple = PtrTupleImpl<IndexSequenceFor<Ts...>, Ts...>;

// -------------------- ValueHolder / ValueTuple (stores values) --------------------
template <size_t I, typename T> struct ValueHolder { T val; };

template <typename IndexSeq, typename... Ts> struct ValueTupleImpl;
template <size_t... Is, typename... Ts>
struct ValueTupleImpl<IndexSequence<Is...>, Ts...> : ValueHolder<Is, Ts>... {
    __device__ ValueTupleImpl(Ts const&... vs) : ValueHolder<Is, Ts>{vs}... {}
    template <size_t I>
    __device__ auto get() -> typename TypeAt<I, Ts...>::type const& {
        return this->template ValueHolder<I, typename TypeAt<I, Ts...>::type>::val;
    }
};
template <typename... Ts> using ValueTuple = ValueTupleImpl<IndexSequenceFor<Ts...>, Ts...>;
// clang-format on

// -------------------- Indices apply (calls functor for each index)
// --------------------
template <size_t N>
struct Indices {
  template <typename F>
  static __device__ void apply(F& f) {
    if constexpr (N > 0) {
      Indices<N - 1>::apply(f);
      f.template operator()<N - 1>();  // pass size_t directly
    }
  }
};

// -------------------- functors --------------------
template <typename Queue, typename PtrTupleT>
struct PopDeviceFunctor {
  Queue* queue;
  PtrTupleT& outs;
  int slot;
  __device__ PopDeviceFunctor(Queue* q, PtrTupleT& o, int s)
      : queue(q), outs(o), slot(s) {}

  template <size_t I>
  __device__ void operator()() {
    auto ptr = outs.template get<I>();
    if (ptr) *ptr = queue->template buffer<I>()[slot];
  }
};

template <typename Queue, typename ValTupleT>
struct PushDeviceFunctor {
  Queue* queue;
  ValTupleT& vals;
  int slot;
  __device__ PushDeviceFunctor(Queue* q, ValTupleT& v, int s)
      : queue(q), vals(v), slot(s) {}

  template <size_t I>
  __device__ void operator()() {
    queue->template buffer<I>()[slot] = vals.template get<I>();
  }
};

#endif

namespace dmt {
constexpr __host__ __device__ size_t alignUp(size_t size, size_t alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

/// Queue stored in unified (managed) memory
/// Allocation uses the driver API cuMemAllocManaged
/// The queue header nad the ring buffer elements are placed in the managed
/// allocation Important: Host and Device must not access concurently
template <typename T>
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
  requires(std::is_trivial_v<T> && std::is_standard_layout_v<T>)
#endif
struct alignas(T) ManagedQueue {
  // metadata: placed in managed memory so both host and device can see it
  // (requires sync)
  int32_t capacity;  // number of elements
  int32_t head;      // index for pop. host-only or protected by synchronization
                     // protocol
  int32_t tail;  // index for push (next free). ring buffer uses modulo capacity
  int32_t count;  // optional count updated with atomic operations

  // data: following this header is properly aligned data.
  // Layout: [ManagedQueue<T> header][T data[capacity]]

  // device helper: obtains pointer to element storage. Works because alignas
  // ensures proper padding
  __host__ __device__ T* data() { return reinterpret_cast<T*>(this + 1); }

  // --- Host-side APIs (no device concurrency) ---
  /// push from host (non-threadsafe w.r.t device access; caller must ensure no
  /// device activity)
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
  /// peek from host: non-threadsafe, does not modify head/tail
  /// @warning debug only
  __host__ bool peekHost(int index, T* out) {
    if (index < 0 || index >= count) return false;
    if (out) {
      int slot = (head + index) % capacity;
      *out = data()[slot];
    }
    return true;
  }

  __host__ bool pushHost(T const& v) {
    if (count >= capacity) return false;
    // simple ring push (no atomics) -- caller must ensure host-only access
    data()[tail] = v;
    tail = (tail + 1) % capacity;
    ++count;
    return true;
  }

  /// pop from host (non-threadsafe w.r.t device access; caller must ensure no
  /// device activity)
  __host__ bool popHost(T* out) {
    if (count <= 0) return false;
    if (out) *out = data()[head];

    head = (head + 1) % capacity;
    --count;
    return true;
  }
#endif

// --- Device-side APIs (safe for many device threads within a kernel) ---
/// Device pushes are implemented with atomic operations so many threads can
/// push, but host must not access concurrently
#if defined(__CUDA_ARCH__)  // necessary as we are not using the nvcc compiler
                            // driver
  __device__ bool pushDevice(T const& v) {
    // reserve an index atomically and compute slot
    int ticket = atomicAdd(&tail, 1);
    int slot = ticket % capacity;

    // if the queue was full, decrement count and mark failure
    int oldCount = atomicAdd(&count, 1);
    if (oldCount >= capacity) {
      // Note: tail rollback is not trivial in
      // high-concurrency context — we accept that tail will have advanced
      // (ticket counter). this might fail if in the meantime some slot gets
      // free
      atomicSub(&count, 1);
      return false;
    }

    // store element into slot
    data()[slot] = v;

    // a device writer may want a release store; CUDA device writes to global
    // memory are visible after store. For simplicity we assume default memory
    // ordering is sufficient for consumer on device or host once
    // synchronization happens.
    return true;
  }

  __device__ bool popDevice(T* out) {
    // reserve head ticket, compute slot
    int ticket = atomicAdd(&head, 1);
    int slot = ticket % capacity;

    int oldCount = atomicSub(&count, 1);
    if (oldCount <= 0) {
      // nothing to pop
      atomicAdd(&count, 1);
      return false;
    }

    if (out) *out = data()[slot];

    return true;
  }
#endif

  // ---- Allocation / deallocation helpers (host only) ----
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
  /// Allocate a ManagedQueue<T> with space for 'cap' elements in managed memory
  /// using driver API cuMemAllocManaged. it is adviseable to use trivial,
  /// standard layout types flags: CU_MEM_ATTACH_GLOBAL or CU_MEM_ATTACH_HOST
  /// (driver API values).
  static inline ManagedQueue<T>* allocateManaged(
      CUDADriverLibrary const& nvApi, int cap, size_t& out_bytes,
      uint32_t flags = CU_MEM_ATTACH_GLOBAL) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "T must be trivially copyable for this queue.");
    size_t const headerSize = sizeof(ManagedQueue<T>);
    // no need to add padding to headerSize if headerSize is aligned to T
    size_t const total = headerSize + static_cast<size_t>(cap) * sizeof(T);

    CUdeviceptr devPtr = 0;
    if (!cudaDriverCall(&nvApi, nvApi.cuMemAllocManaged(&devPtr, total, flags)))
      return nullptr;
    auto* q = std::bit_cast<ManagedQueue<T>*>(devPtr);
    new (q) ManagedQueue<T>();
    q->capacity = cap;
    q->head = 0;
    q->tail = 0;
    q->count = 0;

    out_bytes = total;
    return q;
  }

  static inline void freeManaged(CUDADriverLibrary const& nvApi,
                                 ManagedQueue<T>* q) {
    auto devPtr = std::bit_cast<CUdeviceptr>(q);
    if (devPtr) {
      CUresult res = nvApi.cuMemFree(devPtr);
      if (Context ctx; res != ::CUDA_SUCCESS && ctx.isValid()) {
        char const* ptr = nullptr;
        nvApi.cuGetErrorString(res, &ptr);
        ctx.error("CUDA Error while freeing queue: {}", std::make_tuple(ptr));
      }
    }
  }
#endif
};

//----------------------------------------------------------
template <typename... Ts>
struct ManagedMultiQueue {
  int32_t capacity;  // number of elements
  int32_t head;      // index for pop. host-only or protected by synchronization
                     // protocol
  int32_t tail;  // index for push (next free). ring buffer uses modulo capacity
  int32_t count;  // optional count updated with atomic operations

  // --- Host-side APIs ---
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)

  // allocate all sub-queues in managed memory
  static inline ManagedMultiQueue<Ts...>* allocateManaged(
      CUDADriverLibrary const& nvApi, int cap, size_t& out_bytes,
      uint32_t flags = CU_MEM_ATTACH_GLOBAL) {
    std::vector<size_t> sizes;
    std::vector<size_t> alignments;
    populateSizesAndAlignments<Ts...>(sizes, alignments);
    size_t const alignment = *std::ranges::max_element(alignments);
    size_t const headerPadded =
        alignUp(sizeof(ManagedMultiQueue<Ts...>), alignment);
    size_t totalSize = headerPadded;
    out_bytes = totalSize;

    for (uint32_t i = 0; i < sizes.size(); ++i) {
      if (i != sizes.size() - 1)
        totalSize += alignUp(sizes[i] * static_cast<size_t>(cap), alignment);
      else
        totalSize += sizes[i] * static_cast<size_t>(cap);
    }

    CUdeviceptr devPtr = 0;
    if (!cudaDriverCall(&nvApi,
                        nvApi.cuMemAllocManaged(&devPtr, totalSize, flags)))
      return nullptr;

    auto* mmq = std::bit_cast<ManagedMultiQueue<Ts...>*>(devPtr);
    new (mmq) ManagedMultiQueue<Ts...>();
    mmq->capacity = cap;
    mmq->head = 0;
    mmq->tail = 0;
    mmq->count = 0;

    return mmq;
  }

  static inline void freeManaged(CUDADriverLibrary const& nvApi,
                                 ManagedMultiQueue<Ts...>* mmq) {
    if (!mmq) return;

    CUdeviceptr devPtr = reinterpret_cast<CUdeviceptr>(mmq);
    CUresult res = nvApi.cuMemFree(devPtr);

    if (Context ctx; res != CUDA_SUCCESS && ctx.isValid()) {
      char const* errStr = nullptr;
      nvApi.cuGetErrorString(res, &errStr);
      ctx.error("CUDA Error while freeing ManagedMultiQueue: {}",
                std::make_tuple(errStr));
    }
  }

  // Compute pointer to i-th buffer (T0, T1, ...) in managed memory
  template <std::size_t Index>
  auto* buffer() {
    std::vector<size_t> sizes;
    std::vector<size_t> alignments;
    populateSizesAndAlignments<Ts...>(sizes, alignments);

    size_t const maxAlignment = *std::ranges::max_element(alignments);
    size_t offset = alignUp(sizeof(ManagedMultiQueue<Ts...>), maxAlignment);

    for (std::size_t i = 0; i < Index; ++i)
      offset += alignUp(sizes[i] * static_cast<size_t>(capacity), maxAlignment);

    return reinterpret_cast<std::tuple_element_t<Index, std::tuple<Ts...>>*>(
        reinterpret_cast<std::byte*>(this) + offset);
  }

  // Helper to iterate over tuple indices
  template <typename Tuple, std::size_t... I>
  void pushHostImpl(Tuple const& values, std::index_sequence<I...>) {
    ((buffer<I>()[tail] = std::get<I>(values)), ...);
  }

  template <typename Tuple, std::size_t... I>
  void popHostImpl(Tuple& out, std::index_sequence<I...>) {
    ((std::get<I>(out) = buffer<I>()[head]), ...);
  }

  bool pushHost(std::tuple<Ts...> const& values) {
    if (count >= capacity) return false;

    pushHostImpl(values, std::index_sequence_for<Ts...>{});
    tail = (tail + 1) % capacity;
    ++count;
    return true;
  }

  bool popHost(std::tuple<Ts...>* out) {
    if (count <= 0) return false;

    if (out) popHostImpl(*out, std::index_sequence_for<Ts...>{});

    head = (head + 1) % capacity;
    --count;
    return true;
  }

  // Helper to copy into tuple at slot
  template <typename Tuple, std::size_t... I>
  void peekHostImpl(Tuple& out, int slot, std::index_sequence<I...>) {
    ((std::get<I>(out) = buffer<I>()[slot]), ...);
  }

  bool peekHost(int index, std::tuple<Ts...>* out) {
    if (index < 0 || index >= count) return false;
    if (out) {
      int slot = (head + index) % capacity;
      peekHostImpl(*out, slot, std::index_sequence_for<Ts...>{});
    }
    return true;
  }
#endif

#if defined(__CUDA_ARCH__)

  // buffer pointer computation
  template <size_t Index>
  __device__ typename TypeAt<Index, Ts...>::type* buffer() {
    size_t sizes[] = {sizeof(Ts)...};
    size_t alignments[] = {alignof(Ts)...};
    size_t maxAlign = alignments[0];
    for (size_t i = 1; i < sizeof...(Ts); ++i)
      if (alignments[i] > maxAlign) maxAlign = alignments[i];

    size_t offset = alignUp(sizeof(ManagedMultiQueue<Ts...>), maxAlign);
    for (size_t i = 0; i < Index; ++i)
      offset += alignUp(sizes[i] * static_cast<size_t>(capacity), maxAlign);

    using T = typename TypeAt<Index, Ts...>::type;
    return reinterpret_cast<T*>(reinterpret_cast<char*>(this) + offset);
  }

  // ------------------ Device push ------------------
  // pushDevice: values passed by const-ref
  template <typename... Us>
  __device__ bool pushDevice(Us const&... values_raw) {
    static_assert(sizeof...(Us) == sizeof...(Ts), "Wrong number of values");

    // pack the actual values into ValueTuple
    ValueTuple<Us...> vals(values_raw...);

    int ticket = atomicAdd(&tail, 1);
    int slot = ticket % capacity;

    int oldCount = atomicAdd(&count, 1);
    if (oldCount >= capacity) {
      atomicSub(&count, 1);
      return false;
    }

    PushDeviceFunctor<ManagedMultiQueue<Ts...>, ValueTuple<Us...>> f(this, vals,
                                                                     slot);
    Indices<sizeof...(Ts)>::apply(f);
    return true;
  }

  // ------------------ Device pop ------------------
  // popDevice: outs are pointers to host-scoped locals (on device call use
  // &localVar)
  template <typename... Us>
  __device__ bool popDevice(Us*... outs_raw) {
    static_assert(sizeof...(Us) == sizeof...(Ts), "Wrong number of outs");

    PtrTuple<Us...> outs(outs_raw...);
    int ticket = atomicAdd(&head, 1);
    int slot = ticket % capacity;

    int oldCount = atomicSub(&count, 1);
    if (oldCount <= 0) {
      atomicAdd(&count, 1);
      return false;
    }

    PopDeviceFunctor<ManagedMultiQueue<Ts...>, PtrTuple<Us...>> f(this, outs,
                                                                  slot);
    Indices<sizeof...(Ts)>::apply(f);
    return true;
  }

#endif
};

}  // namespace dmt

// TODO move elsewhere: Payloads and params for each thing

namespace dmt {

struct RaygenPayload {
  // ox, oy, oz
  // dx, dy, dz
  // sample weight
  ManagedMultiQueue<float, float, float, float, float, float, float>* mmq;
};

struct RaygenParams {
  RaygenPayload rayPayload;
  int32_t px;
  int32_t py;
  int32_t sampleIndex;
};
}  // namespace dmt
