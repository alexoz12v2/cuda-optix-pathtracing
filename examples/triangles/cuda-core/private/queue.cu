#include "queue.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ unsigned pushAggGMEM(void const* input, int inputSize,
                                int* reserve_back, int* publish_back,
                                int* queue_open, int* front, void* queue,
                                int queueCapacity) {
  assert(!(reinterpret_cast<intptr_t>(queue) & 0xf));  // 16 byte aligned
  assert((queueCapacity & (queueCapacity - 1)) == 0);
  cg::coalesced_group const g = cg::coalesced_threads();
  int const inputCount = g.size();
  int actualCount = inputCount;
  // best effert to estimate current capacity
  {
    int const currentBack = atomicAdd(reserve_back, 0);
    int const currentFront = atomicAdd(front, 0);
    int const used =
        (currentBack - currentFront + queueCapacity) & (queueCapacity - 1);
    int const freePlaces = queueCapacity - 1 - used;
    actualCount = min(freePlaces, actualCount);
  }

  // reservation
  int base = 0;
  if (g.thread_rank() == 0) {
#if __CUDA_ARCH__ >= 700
    base = __nv_atomic_add(reserve_back, actualCount, __NV_ATOMIC_ACQ_REL,
                           __NV_THREAD_SCOPE_DEVICE);
#else
    base = atomicAdd(reserve_back, actualCount);
    // volatile → compiler cannot remove or move it
    // "memory" clobber → compiler assumes memory is affected
    // ensure queue is in its last up-to-date state
    asm volatile("membar.gl;" ::: "memory");
#endif
  }
  base = g.shfl(base, 0);

  int const idx = base + g.thread_rank();
  int const slot = idx & (queueCapacity - 1);

  // probably not needed since we locked the queue?
  bool const pushed = g.thread_rank() < actualCount;
  if (pushed) {
    // all threads are submitting their own element, which is in LMEM
    memcpy(static_cast<uint8_t*>(queue) + slot * inputSize, input, inputSize);
  }

  // need to be visible to consumers
  if (g.thread_rank() == 0) {
    // asm volatile("membar.gl;" ::: "memory");
    __threadfence();
    // wait until it is our turn to publish
    while (atomicAdd(publish_back, 0) < base) {
      // spin
    }
    atomicAdd(publish_back, actualCount);
    atomicExch(queue_open, 1);
  }

  // last time, everyone gets together
  g.sync();

  // compute warp uniform return value (mask of who did the push successfully)
  unsigned const mask = g.ballot(pushed);
#ifndef NDEBUG
  if (g.thread_rank() == 0) {
    // A contiguous LSB mask looks like: 00011111
    // Adding 1 flips it to: 00100000
    // ANDing them gives 0
    // Any hole breaks this property
    // no holes: must be 0b000...011...1
    assert((mask & (mask + 1)) == 0);
  }
  g.sync();
#endif
  return mask;
}

__device__ unsigned popAggGMEM(void* output, int outputSize, int* publish_back,
                               int* queue_open, int* front, void* queue,
                               int queueCapacity) {
  assert(!(reinterpret_cast<intptr_t>(queue) & 0xf));  // 16 byte aligned
  assert((queueCapacity & (queueCapacity - 1)) == 0);

  cg::coalesced_group const g = cg::coalesced_threads();
  // block until first elements are produced
  if (g.thread_rank() == 0) {
    while (atomicAdd(queue_open, 0) == 0) {
      // spin
    }
  }
  g.sync();

  // will this ensure that, between 2 consumers warps, only one is allowed to
  // proceed?
  int base = 0;
  int claimed = 0;
  if (g.thread_rank() == 0) {
    // acquire semantics
    __threadfence();
    int const pub = atomicAdd(publish_back, 0);
    int const fr = atomicAdd(front, 0);
    int const avail = (pub - fr + queueCapacity) & (queueCapacity - 1);

    claimed = min(avail, g.size());
    if (claimed > 0) {
      base = atomicCAS(front, fr, fr + claimed);
      if (base != fr) {  // another warp got here before me. quit. TODO retry
        claimed = 0;
      }
    }
  }

  claimed = g.shfl(claimed, 0);
  base = g.shfl(base, 0);

  if (claimed == 0) {
    return 0;
  }

  int const idx = base + g.thread_rank();
  int const slot = idx & (queueCapacity - 1);

  // warp-coalesced copy from queue to output
  bool const popped = g.thread_rank() < claimed;
  if (popped) {
    // all threads are submitting their own element, which is in LMEM
    memcpy(output, static_cast<uint8_t*>(queue) + slot * outputSize,
           outputSize);
  }

  g.sync();

  // compute warp uniform return value
  unsigned const mask = g.ballot(popped);
#ifndef NDEBUG
  if (g.thread_rank() == 0) {
    // A contiguous LSB mask looks like: 00011111
    // Adding 1 flips it to: 00100000
    // ANDing them gives 0
    // Any hole breaks this property
    // no holes: must be 0b000...011...1
    assert((mask & (mask + 1)) == 0);
  }
  g.sync();
#endif
  return mask;
}

__device__ int queueAvail(int* back, int* front, int queueCapacity) {
  cg::coalesced_group const g = cg::coalesced_threads();
  int avail = 0;
  if (g.thread_rank() == 0) {
    int const pub = atomicAdd(back, 0);
    int const fr = atomicAdd(front, 0);
    avail = (pub - fr + queueCapacity) & (queueCapacity - 1);
  }
  g.sync();
  g.shfl(avail, 0);
  return avail;
}