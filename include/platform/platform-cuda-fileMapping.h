#pragma once

#include "dmtmacros.h"

#include <utility>
#include <string_view>

// WHy File Mapping? https://stackoverflow.com/questions/15041016/overlapped-io-or-file-mapping

namespace dmt::os {
    /**
     * Class which encapsulates the capability to open a file and read it from a device supporting UVA (__managed__)
     * This class is supposed to be *Warp Local*. During kernel execution, an array with `warpCount` length of this class
     * should be allocated.
     * The requested file should be indicated with a UTF-8 encoded path, alongside with a *Chunksize*.
     * The Chunk Buffer, aligned with the granularity of the system and multiple of such granularity, is subdivided in
     * small pieces of 16 bytes of memory, belonging to lane (`pieceIndex % warpSize`)
     * Upon device request (with any `__activemask`, ideally), if the buffer was already mapped and fetched the requested
     * chunk, then the Warp can be serviced immediately. Otherwise, the warp needs to be put to sleep and set some kind of
     * atomic shared state (eg an int) to signal to the host that the Device is waiting for a memory mapping.
     * This means that, on the host side, there should be a thread listening for device reqeusts, and be ready to service them
     * @note we cannot use `MapViewOfFileEx` by supplying `__managed__` memory, we cannot avoid a memory copy,
     * we need to map the file and propagate memory to device memory device finishes to use the chunk (basically when it requests
     * an unmap)
     * @note for Windows, the chunk should be multiple of 2MB aligned to a 64KB boundary, I think?
     * @note while the mapped buffer is mamaged by this class, the pointer to device mmeory whose size should be equal
     * to the reqeusted chunk size is supplied externally and owned by this class for its whole lifetime. It doesn't free it
     * @note The `chunkSize` is subdivided in blocks of `warpSize * 16 B`, such that each lane writes to the 16 B + i * warpSize * 16 B
     */
    class CudaFileMapping
    {
    public:
        static constexpr uint32_t implSize = 64;
        /**
         * if `create` is true and the file on the path doesn't exist, then a file with size equal to the chunksize should be
         * created. Needs to be UTF-8 string (NOT null terminated)
         *
         * @note For now, tbis function is allowed to allocated memory, in the future it should take a memory resource or use
         * the context stack allocator
         * The `chunksize` must be a multiple of the allocation granularity, divisible by 16 and by the WarpSize
         */
        DMT_CPU CudaFileMapping(std::string_view _fileName, uint32_t _chunkSize, bool _create, void* _target);
        CudaFileMapping(CudaFileMapping const&)                = delete;
        CudaFileMapping(CudaFileMapping&&) noexcept            = delete;
        CudaFileMapping& operator=(CudaFileMapping const&)     = delete;
        CudaFileMapping& operator=(CudaFileMapping&&) noexcept = delete;
        DMT_CPU ~CudaFileMapping();

    public:
        DMT_GPU void requestChunk(int32_t _chunkIndex, uintptr_t _cudaStream = 0);
        DMT_GPU void signalCompletion();
        DMT_CPU bool requestedChunk() const;
        DMT_CPU void signalChunkLoaded();
        DMT_CPU void waitForCompletion();

    public:
        inline DMT_CPU int32_t* chunky() { return &m_chunkReady; }

    public:
        void* target;

    private:
        size_t m_fileSize = 0;
        alignas(std::max_align_t) unsigned char m_impl[implSize]{};
        uint32_t m_chunkSize;
        // TODO store a job handle to the CPU thread listener, and wait on it in the destructor

        /**
         * Stuff manipulated with `std::atomic_ref<int32_t>` on the host,
         * and with `cuda::atomic_ref<int32_t, cuda::std::thread_scope_system>` on device code
         * @warning requires `cudaDeviceProp::concurrentManagedAccess`
         * Actually, my GPU doesn't support this, so ew need to manually synchronize using events
         */
        int32_t m_chunkRequest = -1; // CUDA warp requests a chunk
        int32_t m_chunkReady   = 0;  // CPU signals ready
        //int32_t m_chunkDoce    = 0;  // CUDA signals done
    };
} // namespace dmt::os