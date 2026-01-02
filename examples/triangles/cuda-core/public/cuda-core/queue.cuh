#ifndef DMT_CUDA_CORE_QUEUE_CUH
#define DMT_CUDA_CORE_QUEUE_CUH

#include "cuda-core/types.cuh"

// Synchronization → PTX instruction mapping
// __threadfence() ≈ membar.gl
// __threadfence_block() ≈ membar.cta
// __threadfence_system() ≈ membar.sys

inline __forceinline__ __device__ int atomicAggInc(int* ptr) {
  namespace cg = cooperative_groups;
  cg::coalesced_group g = cg::coalesced_threads();
  int prev = 0;

  // elect the first active thread to perform atomic add
  if (g.thread_rank() == 0) {
    prev = atomicAdd(ptr, g.size());
  }

  // broadcast previous value within the warp
  // and add each active thread’s rank to it
  prev = g.thread_rank() + g.shfl(prev, 0);
  return prev;
}

inline __forceinline__ __device__ int atomicAggDec(int* ptr) {
  namespace cg = cooperative_groups;
  cg::coalesced_group g = cg::coalesced_threads();
  int prev = 0;

  // elect the first active thread to perform atomic add
  if (g.thread_rank() == 0) {
    prev = atomicSub(ptr, g.size());
  }

  // broadcast previous value within the warp
  // and add each active thread’s rank to it
  prev = g.thread_rank() - g.shfl(prev, 0);
  return prev;
}

inline __forceinline__ __device__ int atomicAggDecSaturate(int* ptr) {
  namespace cg = cooperative_groups;
  cg::coalesced_group g = cg::coalesced_threads();
  int prev = 0;
  int dec = 0;

  if (g.thread_rank() == 0) {
    // cap decrement to remaining items
    int const old = atomicAdd(ptr, 0);
    dec = min(old, g.size());
    if (dec > 0) {
      prev = atomicSub(ptr, dec);
    }
  }

  prev = g.shfl(prev, 0);
  dec = g.shfl(dec, 0);

  // Each lane gets its own ticket (or -1)
  int const my = prev - 1 - g.thread_rank();
  return (g.thread_rank() < dec) ? my : -1;
}

__device__ unsigned pushAggGMEM(void const* input, int inputSize,
                                int* reserve_back, int* publish_back,
                                int* queue_open, int* front, void* queue,
                                int queueCapacity);
__device__ unsigned popAggGMEM(void* output, int outputSize, int* publish_back,
                               int* queue_open, int* front, void* queue,
                               int queueCapacity);

// now back starts from 0 and front starts from 0
// Queue is empty → back == front
// Queue is full  → (back + 1) % queueCapacity == front % queueCapacity

template <typename T>
  requires std::is_trivial_v<T> && std::is_standard_layout_v<T>
struct QueueGMEM {
  T* queue;
  int* front;
  int* reserve_back;
  int* publish_back;
  int* queue_open;
  int queueCapacity;

  __device__ __forceinline__ unsigned push(T const* elem) {
    return pushAggGMEM(elem, sizeof(T), reserve_back, publish_back, queue_open,
                       front, queue, queueCapacity);
  }
  __device__ __forceinline__ unsigned pop(T* elem) {
    return popAggGMEM(elem, sizeof(T), publish_back, queue_open, front, queue,
                      queueCapacity);
  }
};

#endif