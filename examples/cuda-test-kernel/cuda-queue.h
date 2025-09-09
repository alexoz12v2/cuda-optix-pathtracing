#pragma once

#include "dmtmacros.h"

#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
    #include "cuda-wrappers/cuda-wrappers-utils.h"
    #include <type_traits>
#endif

namespace dmt {

    /// Queue stored in unified (managed) memory
    /// Allocation uses the driver API cuMemAllocManaged
    /// The queue header nad the ring buffer elements are placed in the managed allocation
    /// Important: Host and Device must not access concurently
    template <typename T>
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
        requires(std::is_trivial_v<T> && std::is_standard_layout_v<T>)
#endif
    struct alignas(T) ManagedQueue
    {
        // metadata: placed in managed memory so both host and device can see it (requires sync)
        int32_t capacity; // number of elements
        int32_t head;     // index for pop. host-only or protected by synchronization protocol
        int32_t tail;     // index for push (next free). ring buffer uses modulo capacity
        int32_t count;    // optional count updated with atomic operations

        // data: following this header is properly aligned data.
        // Layout: [ManagedQueue<T> header][T data[capacity]]

        // device helper: obtains pointer to element storage. Works because alignas ensures proper padding
        __host__ __device__ T* data() { return reinterpret_cast<T*>(this + 1); }

        // --- Host-side APIs (no device concurrency) ---
        /// push from host (non-threadsafe w.r.t device access; caller must ensure no device activity)
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
        __host__ bool pushHost(T const& v)
        {
            if (count >= capacity)
                return false;
            // simple ring push (no atomics) -- caller must ensure host-only access
            data()[tail] = v;
            tail         = (tail + 1) % capacity;
            ++count;
            return true;
        }

        /// pop from host (non-threadsafe w.r.t device access; caller must ensure no device activity)
        __host__ bool popHost(T* out)
        {
            if (count <= 0)
                return false;
            if (out)
                *out = data()[head];

            head = (head + 1) % capacity;
            --count;
            return true;
        }
#endif

// --- Device-side APIs (safe for many device threads within a kernel) ---
/// Device pushes are implemented with atomic operations so many threads can push, but host must not access concurrently
#if defined(__CUDA_ARCH__) // necessary as we are not using the nvcc compiler driver
        __device__ bool pushDevice(T const& v)
        {
            // reserve an index atomically and compute slot
            int ticket = atomicAdd(&tail, 1);
            int slot   = ticket % capacity;

            // if the queue was full, decrement count and mark failure
            int oldCount = atomicAdd(&count, 1);
            if (oldCount >= capacity)
            {
                // Note: tail rollback is not trivial in
                // high-concurrency context — we accept that tail will have advanced (ticket counter).
                // this might fail if in the meantime some slot gets free
                atomicSub(&count, 1);
                return false;
            }

            // store element into slot
            data()[slot] = v;

            // a device writer may want a release store; CUDA device writes to global memory are visible after store.
            // For simplicity we assume default memory ordering is sufficient for consumer on device or host once synchronization happens.
            return true;
        }

        __device__ bool popDevice(T* out)
        {
            // reserve head ticket, compute slot
            int ticket = atomicAdd(&head, 1);
            int slot   = ticket % capacity;

            int oldCount = atomicSub(&count, 1);
            if (oldCount <= 0)
            {
                // nothing to pop
                atomicAdd(&count, 1);
                return false;
            }

            if (out)
                *out = data()[slot];

            return true;
        }
#endif

        // ---- Allocation / deallocation helpers (host only) ----
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
        /// Allocate a ManagedQueue<T> with space for 'cap' elements in managed memory using driver API cuMemAllocManaged.
        /// it is adviseable to use trivial, standard layout types
        /// flags: CU_MEM_ATTACH_GLOBAL or CU_MEM_ATTACH_HOST (driver API values).
        static inline ManagedQueue<T>* allocateManaged(CUDADriverLibrary const& nvApi,
                                                       int                      cap,
                                                       size_t&                  out_bytes,
                                                       uint32_t                 flags = CU_MEM_ATTACH_HOST)
        {
            static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable for this queue.");
            size_t const headerSize = sizeof(ManagedQueue<T>);
            size_t const alignment  = alignof(T);
            // no need to add padding to headerSize if headerSize is aligned to T
            size_t const total = headerSize + static_cast<size_t>(cap) * sizeof(T);

            CUdeviceptr devPtr = 0;
            if (!cudaDriverCall(&nvApi, nvApi.cuMemAllocManaged(&devPtr, total, flags)))
                return nullptr;
            auto* q = std::bit_cast<ManagedQueue<T>*>(devPtr);
            new (q) ManagedQueue<T>();
            q->capacity = cap;
            q->head     = 0;
            q->tail     = 0;
            q->count    = 0;

            out_bytes = total;
            return q;
        }

        static inline void freeManaged(CUDADriverLibrary const& nvApi, ManagedQueue<T>* q)
        {
            auto devPtr = std::bit_cast<CUdeviceptr>(q);
            if (devPtr)
            {
                CUresult res = nvApi.cuMemFree(devPtr);
                if (Context ctx; res != ::CUDA_SUCCESS && ctx.isValid())
                {
                    char const* ptr = nullptr;
                    nvApi.cuGetErrorString(res, &ptr);
                    ctx.error("CUDA Error while freeing queue: {}", std::make_tuple(ptr));
                }
            }
        }
#endif
    };
} // namespace dmt
