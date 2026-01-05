#include "queue.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define OTHER_THING 0

#if OTHER_THING
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
    base = atomicAdd(reserve_back, actualCount);
    // volatile → compiler cannot remove or move it
    // "memory" clobber → compiler assumes memory is affected
    // ensure queue is in its last up-to-date state
    asm volatile("membar.gl;" ::: "memory");
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
#  ifndef NDEBUG
  if (g.thread_rank() == 0) {
    // A contiguous LSB mask looks like: 00011111
    // Adding 1 flips it to: 00100000
    // ANDing them gives 0
    // Any hole breaks this property
    // no holes: must be 0b000...011...1
    assert((mask & (mask + 1)) == 0);
  }
  g.sync();
#  endif
  return mask;
}
#else
__device__ unsigned pushAggGMEM(void const* input, int inputSize,
                                int* reserve_back, int* publish_back,
                                int* queue_open, int* front, void* queue,
                                int queueCapacity) {
  cg::coalesced_group g = cg::coalesced_threads();
  int const laneId = g.thread_rank();

  // 1. Unified Reservation
  unsigned base = 0;
  int actualCount = 0;

  if (laneId == 0) {
    // TODO: these two should be read _at the same time_
    unsigned const snapFront = *front;
    unsigned const snapBack = *reserve_back;
    int const freePlaces = queueCapacity - (snapBack - snapFront);

    actualCount = max(0, min(static_cast<int>(g.size()), freePlaces));
    if (actualCount > 0) {
      base = atomicAdd(reserve_back, actualCount);
    }
  }

  actualCount = g.shfl(actualCount, 0);
  base = g.shfl(base, 0);

  bool const pushed = laneId < actualCount;
  if (pushed) {
    unsigned const slot = (base + laneId) & (queueCapacity - 1);
    memcpy(static_cast<uint8_t*>(queue) + slot * inputSize, input, inputSize);
  }

  // 2. Safe Publishing (Ordered)
  if (actualCount > 0 && laneId == 0) {
    __threadfence();  // Ensure memcpy is visible before updating publish_back

    // DANGER: If you keep the ordered spin-lock, you MUST use
    // a "yield" or ensure high occupancy.
    // Better: Use an array of "ready" flags for each slot.
    while (*publish_back != base) {
      // TODO Optional: __nanosleep() or pascal_fixed_sleep()
    }
    atomicAdd(publish_back, actualCount);
  }

  g.sync();  // All threads must hit this
  return g.ballot(pushed);
}
#endif

#if OTHER_THING
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
  return g.ballot(popped);
}
#else
__device__ unsigned popAggGMEM(void* output, int outputSize, int* publish_back,
                               int* queue_open, int* front, void* queue,
                               int queueCapacity) {
  cg::coalesced_group g = cg::coalesced_threads();
  int const laneId = g.thread_rank();
  unsigned const mask = queueCapacity - 1;

  unsigned base = 0;
  int claimed = 0;

  if (laneId == 0) {
    // TODO: these two should be read _at the same time_
    unsigned const snapPub = *publish_back;
    unsigned const snapFront = *front;

    int const avail = snapPub - snapFront;
    claimed = max(0, min(static_cast<int>(g.size()), avail));

    if (claimed > 0) {
      base = atomicAdd(front, claimed);
    }
  }

  claimed = g.shfl(claimed, 0);
  base = g.shfl(base, 0);

  bool const popped = laneId < claimed;
  if (popped) {
    unsigned const slot = (base + laneId) & mask;
    memcpy(output, static_cast<uint8_t*>(queue) + slot * outputSize,
           outputSize);
  }

  g.sync();  // Ensure no threads exit before the group-wide operation is done
  return g.ballot(popped);
}
#endif

// number of available pushes (empty slots)
// (TODO if consecutive, attempt vector access)
__device__ int queueConsumerUsedCount(int* publish_back, int* front,
                                      int capacity) {
  // if GMEM, this is a broadcast access
  int const h = *front;
  int const t = *publish_back;
  return t - h;
}

// number of available pops (full slots, already published)
__device__ int queueProducerFreeCount(int* reserve_back, int* front,
                                      int capacity) {
  // if GMEM, this is a broadcast access
  int const h = *front;
  int const t = *reserve_back;
  return capacity - (t - h);
}
