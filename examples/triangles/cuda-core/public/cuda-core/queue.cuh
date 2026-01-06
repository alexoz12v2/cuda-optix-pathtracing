#ifndef DMT_CUDA_CORE_QUEUE_CUH
#define DMT_CUDA_CORE_QUEUE_CUH

#include "common_math.cuh"
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

  // Helper: Load Cache Volatile (Pascal+)
  // Reads directly from L2/Memory, bypassing potentially stale L1
  __device__ __forceinline__ int load_cv(int* addr) {
    int val;
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
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

  // -------------------------------------------------------------------------
  // Optimized Push (Producer)
  // -------------------------------------------------------------------------
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
        int h = q.load_cv(q.head);  // Volatile load
        int t = q.load_cv(q.tail);  // Volatile load

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

        // Backoff if full to reduce GMEM contention
        q.sleep_ns(200);
      }
    }

    // Broadcast reservation to the warp
    count = g.shfl(count, 0);
    ticket = g.shfl(ticket, 0);

    // 2. Write Data
    bool active = lane < count;
    if (active) {
      int idx = (ticket + lane) & (q.capacity - 1);

      // A. Spin-wait for slot to be EMPTY
      // We use volatile load (cv) instead of atomicAdd for the check
      int backoff = 32;
      while (q.load_cv(&q.states[idx]) != SLOT_EMPTY) {
        q.sleep_ns(backoff);
        backoff = min(backoff * 2, 1024);  // Exponential backoff
      }

      // B. Write Payload (Coalesced access now!)
      q.buffer[idx] = *input;

      // C. Memory Barrier
      // Ensure data is visible before we flip state to FULL
      __threadfence();

      // D. Release Slot
      // Use atomicExch (or atomicStore) to publish availability
      atomicExch(&q.states[idx], SLOT_FULL);
    }

    return g.ballot(active);
  }

  __device__ unsigned queuePop(T* output) {
    namespace cg = cooperative_groups;
    DeviceQueue<T>& q = *this;
    cg::coalesced_group g = cg::coalesced_threads();
    int lane = g.thread_rank();

    unsigned ticket = 0;
    int count = 0;

    if (lane == 0) {
      int h = q.load_cv(q.head);
      int t = q.load_cv(q.tail);

      int avail = t - h;

      // We only reserve what is visibly available in the counters.
      // Even if avail > 0, the slot state might not be FULL yet (producer
      // latency), but that is handled by the spin-wait in step 2.
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
      while (q.load_cv(&q.states[idx]) != SLOT_FULL) {
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
};  // ---------- end class

template <typename T>
inline __host__ void freeQueue(DeviceQueue<T>& q) {
  CUDA_CHECK(cudaFree(q.buffer));
  CUDA_CHECK(cudaFree(q.states));
  CUDA_CHECK(cudaFree(q.head));
  CUDA_CHECK(cudaFree(q.tail));
}

template <typename T>
void __host__ initQueue(DeviceQueue<T>& q, int capacity) {
  q.capacity = capacity;
  cudaMalloc(&q.buffer, sizeof(T) * capacity);
  cudaMalloc(&q.states, sizeof(int) * capacity);
  cudaMalloc(&q.head, sizeof(int));
  cudaMalloc(&q.tail, sizeof(int));

  cudaMemset(q.states, 0, sizeof(int) * capacity);  // All SLOT_EMPTY
  cudaMemset(q.head, 0, sizeof(int));
  cudaMemset(q.tail, 0, sizeof(int));
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

  // Helper: Load Cache Volatile (Pascal+)
  // Reads directly from L2/Memory, bypassing potentially stale L1
  __device__ __forceinline__ int load_cv(int* addr) {
    int val;
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
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

    int my_slot = -1;
    int target_word = -1;

    // 1. Leader finds candidate word
    if (lane == 0) {
      int need = g.size();
      for (int i = 0; i < mask_words; ++i) {
        unsigned int m = load_cv((int*)&bitmask[i]);
        if ((32 - __popc(m)) >= need) {
          target_word = i;
          break;
        }
      }
    }

    target_word = g.shfl(target_word, 0);
    if (target_word < 0) return -1;

    // 2. Cooperative claim with bounded retry
    for (int attempt = 0; attempt < 8; ++attempt) {
      unsigned int old_mask = load_cv((int*)&bitmask[target_word]);
      int free = 32 - __popc(old_mask);
      if (free < g.size()) return -1;

      // Determine ordinal among allocators
      int my_ord = g.thread_rank();  // all lanes participate

      // Find my_ord-th zero bit
      unsigned int my_bit = 0;
      int seen = 0;
      for (int b = 0; b < 32; ++b) {
        if (!(old_mask & (1u << b))) {
          if (seen == my_ord) {
            my_bit = 1u << b;
            break;
          }
          ++seen;
        }
      }

      // Build combined mask
      unsigned int combined = 0;
      for (int i = 0; i < g.size(); ++i) {
        combined |= g.shfl(my_bit, i);
      }

      unsigned int prev =
          atomicCAS(&bitmask[target_word], old_mask, old_mask | combined);

      if (prev == old_mask) {
        if (my_bit) {
          my_slot = target_word * 32 + (__ffs(my_bit) - 1);
        }
        return my_slot;
      }
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