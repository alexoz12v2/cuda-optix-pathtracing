#include "queue.cuh"

#include "common_math.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ unsigned pushAggGMEM(void const* input, int inputSize,
                                int* reserve_back, int* publish_back,
                                int* queue_open, int* front, void* queue,
                                int queueCapacity) {
  cg::coalesced_group g = cg::coalesced_threads();
  int const laneId = g.thread_rank();

  unsigned base = 0;
  int actualCount = 0;

  if (laneId == 0) {
    int volatile const snapFront = *front;
    int volatile const snapBack = *reserve_back;

    int const used = (snapBack - snapFront) & (queueCapacity - 1);
    int const freePlaces = queueCapacity - used;
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

  // 2. Safe Publishing (Unordered)
  if (actualCount > 0 && laneId == 0) {
    __threadfence();

#if 1
    // FIX: Force reload from memory
    // volatile int* tells compiler: "This value changes outside your control"
    volatile int* v_pub = publish_back;

    while (*v_pub != base) {
      // Ideally use a sleep that enforces memory reload, but volatile handles
      // the load.
      // __nanosleep(64) is preferred on Ampere+
      pascal_fixed_sleep(64);
    }
#endif
    atomicAdd(publish_back, actualCount);
  }

  g.sync();
  return g.ballot(pushed);
}

__device__ unsigned popAggGMEM(void* output, int outputSize, int* publish_back,
                               int* queue_open, int* front, void* queue,
                               int queueCapacity) {
  cg::coalesced_group const g = cg::coalesced_threads();
  int const laneId = g.thread_rank();
  unsigned const mask = queueCapacity - 1;

  unsigned base = 0;
  int claimed = 0;

  if (laneId == 0) {
    // FIX: Force reload.
    // If we read a cached '0' once, we might never see the new data.
    int volatile const snapPub = *publish_back;
    int volatile const snapFront = *front;

    int const avail = (snapPub - snapFront) & (queueCapacity - 1);
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

// number of available pushes (empty slots)
// (TODO if consecutive, attempt vector access)
__device__ int queueConsumerUsedCount(int* publish_back, int* front,
                                      int capacity) {
  // if GMEM, this is a broadcast access
  int const h = *front;
  int const t = *publish_back;
  return (t - h) & (capacity - 1);
}

// number of available pops (full slots, already published)
__device__ int queueProducerFreeCount(int* reserve_back, int* front,
                                      int capacity) {
  // if GMEM, this is a broadcast access
  int const h = *front;
  int const t = *reserve_back;
  return capacity - (t - h) & (capacity - 1);
}
