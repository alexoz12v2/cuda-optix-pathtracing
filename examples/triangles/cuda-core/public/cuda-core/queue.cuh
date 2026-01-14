#ifndef DMT_CUDA_CORE_QUEUE_CUH
#define DMT_CUDA_CORE_QUEUE_CUH

#include "common_math.cuh"
#include "cuda-core/types.cuh"

// Synchronization → PTX instruction mapping
// __threadfence() ≈ membar.gl
// __threadfence_block() ≈ membar.cta
// __threadfence_system() ≈ membar.sys

// Helper: Load Cache Volatile (Pascal+)
// Reads directly from L2/Memory, bypassing potentially stale L1
template <typename T>
  requires std::is_same_v<int, T> || std::is_same_v<float, T> ||
           std::is_same_v<unsigned, T>
__device__ __forceinline__ T load_cv(T* addr) {
  T val;
  if constexpr (std::is_same_v<int, T>) {
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(val) : "l"(addr));
  } else if constexpr (std::is_same_v<float, T>) {
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(val) : "l"(addr));
  } else if constexpr (std::is_same_v<unsigned, T>) {
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(val) : "l"(addr));
  }
  return val;
}

__forceinline__ __device__ int atomicAggInc(int* ptr) {
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

__forceinline__ __device__ int atomicAggDec(int* ptr) {
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

__forceinline__ __device__ int atomicAggDecSaturate(int* ptr) {
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

__device__ int queueConsumerUsedCount(int* publish_back, int* front,
                                      int capacity);
__device__ int queueProducerFreeCount(int* reserve_back, int* front,
                                      int capacity);

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

  static __host__ QueueGMEM create(int capacity) {
    int init[4] = {0, 0, 0, 0};
    QueueGMEM<T> oqueue{};
    oqueue.queueCapacity = capacity;
    CUDA_CHECK(cudaMalloc(&oqueue.queue, sizeof(T) * oqueue.queueCapacity));
    CUDA_CHECK(cudaMalloc(&oqueue.front, sizeof(int) * 4));
    CUDA_CHECK(cudaMemcpy(oqueue.front, init, sizeof(int) * 4,
                          cudaMemcpyHostToDevice));
    oqueue.reserve_back = oqueue.front + 1;
    oqueue.publish_back = oqueue.front + 2;
    oqueue.queue_open = oqueue.front + 3;
    return oqueue;
  }

  static __host__ void free(QueueGMEM<T>& queue) {
    CUDA_CHECK(cudaFree(queue.queue));
    CUDA_CHECK(cudaFree(queue.front));
  }

  __device__ __forceinline__ int producerFreeCount() const {
    return queueProducerFreeCount(reserve_back, front, queueCapacity);
  }

  __device__ __forceinline__ int consumerUsedCount() const {
    return queueConsumerUsedCount(publish_back, front, queueCapacity);
  }

  __device__ __forceinline__ unsigned push(T const* elem) {
    return pushAggGMEM(elem, sizeof(T), reserve_back, publish_back, queue_open,
                       front, queue, queueCapacity);
  }
  __device__ __forceinline__ unsigned pop(T* elem) {
    return popAggGMEM(elem, sizeof(T), publish_back, queue_open, front, queue,
                      queueCapacity);
  }
};

// ---------------------------------------------------------------------------
// Slotted queue
// ---------------------------------------------------------------------------

// Enum for readable states
enum SlotState : int {
  SLOT_EMPTY = 0,
  SLOT_WRITING = 1,  // Optional: useful for debugging hanging threads
  SLOT_FULL = 2,
  SLOT_READING = 3  // Optional
};

template <typename T>
struct DeviceQueue {
  T* buffer;     // Data array (Structure of Arrays layout)
  int* states;   // State array
  int* head;     // Consumer index
  int* tail;     // Producer index
  int capacity;  // Size of arrays (must be power of 2)

  // Warning: Potential Deadlock
  __device__ __forceinline__ void spinPush(const T* input) {
    bool pushed = false;
    while (!pushed) {
      int const pushersMask = __activemask();
      int const coalescedMask = queuePush<true>(&input);
      if (!(coalescedMask & (1 << getCoalescedLaneId(pushersMask)))) {
        pascal_fixed_sleep(64);
      } else {
        pushed = true;
      }
    }
  }

  // Helper: Nano Sleep (Pascal+)
  // Reduces memory contention during spin-waiting
  __device__ __forceinline__ void sleep_ns(int ns) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    pascal_fixed_sleep(ns);
#endif
  }

  // Optimized Push (Producer)
  template <bool Loop>
  __device__ unsigned queuePush(const T* input) {
    namespace cg = cooperative_groups;
    DeviceQueue<T>& q = *this;
    cg::coalesced_group g = cg::coalesced_threads();
    int lane = g.thread_rank();

    unsigned ticket = 0;
    int count = 0;

    // 1. Cooperative Reservation
    if (lane == 0) {
      // Retry loop to prevent massive overshoot when queue is full
      while (true) {
        int h = load_cv(q.head);  // Volatile load
        int t = load_cv(q.tail);  // Volatile load

        // Check usage using wrapping arithmetic
        // Note: This snapshot might be slightly stale, but prevents
        // blindly incrementing tail when queue is obviously full.
        int used = t - h;
        if (used < q.capacity) {
          // Determine how many this warp can push
          count = min((int)g.size(), q.capacity - used);

          // ATOMIC RESERVATION
          // This is the commit point.
          ticket = atomicAdd(q.tail, count);
          break;
        }

        // if places not available and there's no active consumer (e.g. no
        // cooperative kernel launch) die immediately
        if constexpr (!Loop) {
          count = 0;
          break;
        }

        // Backoff if full to reduce GMEM contention
        q.sleep_ns(200);
      }
    }

    // Broadcast reservation to the warp
    count = g.shfl(count, 0);
    ticket = g.shfl(ticket, 0);

    // 2. Write Data
    bool active = lane < count;
    int const idx = (ticket + lane) & (q.capacity - 1);
    if (active) {
      // A. Spin-wait for slot to be EMPTY
      // We use volatile load (cv) instead of atomicAdd for the check
      int backoff = 32;
      while (load_cv(&q.states[idx]) != SLOT_EMPTY) {
        if constexpr (!Loop) {
          active = false;
          break;
        }
        q.sleep_ns(backoff);
        backoff = min(backoff * 2, 1024);  // Exponential backoff
      }
    }

    if (active) {
      // B. Write Payload (TODO Coalesced?)
      q.buffer[idx] = *input;

      // C. Memory Barrier
      // Ensure data is visible before we flip state to FULL
      __threadfence();

      // D. Release Slot
      atomicExch(&q.states[idx], SLOT_FULL);
    }

    return g.ballot(active);
  }

  template <bool Loop>
  __device__ unsigned queuePop(T* output) {
    namespace cg = cooperative_groups;
    DeviceQueue<T>& q = *this;
    cg::coalesced_group g = cg::coalesced_threads();
    int lane = g.thread_rank();

    unsigned ticket = 0;
    int count = 0;

    if (lane == 0) {
      int h = load_cv(q.head);
      int t = load_cv(q.tail);

      int avail = t - h;

      // We only reserve what is visibly available in the counters.
      // Even if avail > 0, the slot state might not be FULL yet (producer
      // latency), but that is handled by the spin-wait in step 2.
      // Note: The above^^^ is valid only in a cooperative kernel or in a
      // situation in which it is guaranteed that there are consumers active (
      // specialize warp id within same block)
      count = min((int)g.size(), avail);

      if (count > 0) {
        ticket = atomicAdd(q.head, count);
      }
    }

    count = g.shfl(count, 0);
    ticket = g.shfl(ticket, 0);

    bool active = lane < count;
    if (active) {
      int idx = (ticket + lane) & (q.capacity - 1);

      // A. Spin-wait for slot to be FULL
      int backoff = 32;
      while (load_cv(&q.states[idx]) != SLOT_FULL) {
        if constexpr (!Loop) {
          active = false;
          break;
        }
        q.sleep_ns(backoff);
        backoff = min(backoff * 2, 1024);
      }

      // B. Read Payload (Coalesced)
      *output = q.buffer[idx];

      // C. Memory Barrier
      __threadfence();

      // D. Free Slot
      atomicExch(&q.states[idx], SLOT_EMPTY);
    }

    return g.ballot(active);
  }

  __device__ bool empty_agg() const {
    namespace cg = cooperative_groups;
    cg::coalesced_group const g = cg::coalesced_threads();
    int const lane = g.thread_rank();

    unsigned localOr = 0;
    // stride over bitmask works
    for (int i = lane; i < capacity; i += g.size()) {
      localOr |= load_cv<int>(&states[i]) != SLOT_EMPTY;
    }
    // coalesced warp reduction
    for (int offset = g.size() >> 1; offset > 0; offset >>= 1) {
      localOr |= g.shfl_down(localOr, offset);
    }
    bool empty = localOr == 0;
    empty = g.shfl(empty, 0);
    return empty;
  }
};  // ---------- end class

template <typename T>
__host__ void freeQueue(DeviceQueue<T>& q) {
  CUDA_CHECK(cudaFree(q.buffer));
  CUDA_CHECK(cudaFree(q.states));
  CUDA_CHECK(cudaFree(q.head));
  CUDA_CHECK(cudaFree(q.tail));
}

template <typename T>
void __host__ initQueue(DeviceQueue<T>& q, int capacity) {
  q.capacity = capacity;
  CUDA_CHECK(cudaMalloc(&q.buffer, sizeof(T) * capacity));
  CUDA_CHECK(cudaMalloc(&q.states, sizeof(int) * capacity));
  CUDA_CHECK(cudaMalloc(&q.head, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&q.tail, sizeof(int)));

  CUDA_CHECK(
      cudaMemset(q.states, 0, sizeof(int) * capacity));  // All SLOT_EMPTY
  CUDA_CHECK(cudaMemset(q.head, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(q.tail, 0, sizeof(int)));
}

// ----------------------------------------------------------------------------
// Device Arena
// ----------------------------------------------------------------------------

template <typename T>
struct DeviceArena {
  T* buffer;
  unsigned int* bitmask;  // Each bit: 0=Free, 1=Occupied
  int capacity;           // Must be multiple of 32
  int mask_words;         // capacity / 32

  // Helper: Find first zero bit in a 32-bit word
  __device__ __forceinline__ int find_free_bit(unsigned int mask) {
    // __ffs (~mask) finds the first 1-bit in the inverse (the first 0)
    return __ffs(~mask) - 1;
  }

  // Helper: Nano Sleep (Pascal+)
  // Reduces memory contention during spin-waiting
  __device__ __forceinline__ void sleep_ns(int ns) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    pascal_fixed_sleep(ns);
#endif
  }

  // Note: Every time you write to a occupied slot, __threadfence()
  // Arena Allocate (Warp Aggregated)
  __device__ int allocate() {
    namespace cg = cooperative_groups;
    cg::coalesced_group g = cg::coalesced_threads();
    int lane = g.thread_rank();

    // 1. Leader finds a candidate word
    int target_word = -1;
    if (lane == 0) {
      for (int i = 0; i < mask_words; ++i) {
        unsigned int m = load_cv((int*)&bitmask[i]);
        if (__popc(~m) >= g.size()) {  // Use ~m to count zeros
          target_word = i;
          break;
        }
      }
    }
    target_word = g.shfl(target_word, 0);
    if (target_word < 0) return -1;

    // 2. Attempt to claim
    for (int attempt = 0; attempt < 8; ++attempt) {
      unsigned int old_mask = load_cv(&bitmask[target_word]);
      unsigned int free_mask = ~old_mask;
      int total_free = __popc(free_mask);

      if (total_free < g.size()) {
        // Not enough space in this word anymore, leader pick new word or exit
        return -1;
      }

      // Each thread finds ONE unique bit from the free_mask
      unsigned int my_bit = 0;
      unsigned int temp_mask = free_mask;

      // This is a common idiom: each thread peels off the 'lane'-th bit
      for (int i = 0; i < lane; ++i) {
        temp_mask &= (temp_mask - 1);  // Clear the lowest set bit
      }
      my_bit = temp_mask & ~(temp_mask - 1);  // Isolate the new lowest set bit

      // Combine all bits found by all threads in the group
      unsigned int combined = 0;
      // Optimization: Instead of shfl loop, use the group's internal bitmask
      // But since bits are positions in the 32-bit word, shfl is necessary
      // unless you use a different approach.
      for (int i = 0; i < g.size(); ++i) {
        combined |= g.shfl(my_bit, i);
      }

      unsigned int prev = -1U;
      if (g.thread_rank() == 0) {
        prev = atomicCAS(&bitmask[target_word], old_mask, old_mask | combined);
      }
      prev = g.shfl(prev, 0);

      if (prev == old_mask) {
        return target_word * 32 + (__ffs(my_bit) - 1);
      }

      // If CAS failed, someone else modified the word. Retry.
      // Bounded sleep to reduce contention
      sleep_ns(10);
    }

    return -1;
  }

  // Arena Free
  __device__ void free_slot(int slot_idx) {
    // TODO aggregated free
    if (slot_idx < 0) return;

    int word_idx = slot_idx / 32;
    int bit_offset = slot_idx % 32;
    unsigned int mask = ~(1 << bit_offset);

    // Atomic Clear: bitmask[word_idx] &= ~(1 << bit_offset)
    atomicAnd(&bitmask[word_idx], mask);
  }

  __device__ bool empty_agg() const {
    namespace cg = cooperative_groups;
    cg::coalesced_group const g = cg::coalesced_threads();
    int const lane = g.thread_rank();

    unsigned localOr = 0;
    // stride over bitmask works
    for (int i = lane; i < mask_words; i += g.size()) {
      localOr |= load_cv((int*)&bitmask[i]);
    }
    // coalesced warp reduction
    for (int offset = g.size() >> 1; offset > 0; offset >>= 1) {
      localOr |= g.shfl_down(localOr, offset);
    }
    bool empty = localOr == 0;
    empty = g.shfl(empty, 0);
    return empty;
  }
};

template <typename T>
void initDeviceArena(DeviceArena<T>& arena, int capacity) {
  assert((capacity % 32) == 0);

  arena.capacity = capacity;
  arena.mask_words = capacity / 32;

  CUDA_CHECK(cudaMalloc(&arena.buffer, capacity * sizeof(T)));
  CUDA_CHECK(cudaMemset(arena.buffer, 0, capacity * sizeof(T)));
  CUDA_CHECK(
      cudaMalloc(&arena.bitmask, arena.mask_words * sizeof(unsigned int)));
  CUDA_CHECK(
      cudaMemset(arena.bitmask, 0, arena.mask_words * sizeof(unsigned int)));
}

template <typename T>
void freeDeviceArena(DeviceArena<T>& arena) {
  if (arena.buffer) {
    CUDA_CHECK(cudaFree(arena.buffer));
  }
  if (arena.bitmask) {
    CUDA_CHECK(cudaFree(arena.bitmask));
  }
  arena.buffer = nullptr;
  arena.bitmask = nullptr;
}

#endif
