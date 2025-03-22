#include "platform-cuda-fileMapping.h"

#include "platform-os-utils.win32.h"

#include <atomic>
#include <bit>
#include <memory>
#include <thread>

#include <cassert>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <Windows.h>

namespace dmt::os {
    /** number of bytes needed to store a UTF-16 LE/BE string from a UTF-8 string (NOT null terminated) */
    static size_t numBytes_utf16le_from_utf8(char const* s, uint32_t numBytes, bool addNulTerminateor)
    {
        size_t l = 0;
        for (uint32_t i = 0; i < numBytes; ++i)
            l += (s[i] - 0x80U >= 0x40) + (s[i] >= 0xf0);
        return (l + addNulTerminateor) << 1;
    }

    struct FileMappingWin32
    {
        HANDLE      hFile;
        HANDLE      hFileMapping;
        void*       currentView;
        cudaEvent_t chunkDoneEvent;
        cudaEvent_t chunkReadyEvent;
    };
    static_assert(sizeof(FileMappingWin32) <= CudaFileMapping::implSize &&
                  alignof(FileMappingWin32) <= alignof(std::max_align_t));

    // TODO take allocator like a stack one
    __host__ CudaFileMapping::CudaFileMapping(std::string_view _fileName, uint32_t _chunkSize, bool _create, void* _target) :
    target(_target),
    m_chunkSize(_chunkSize)
    {
        // multiple of address allocation granularity
        static constexpr uint32_t _64KB = 64 * 1024;
        assert((m_chunkSize & (_64KB - 1)) == 0);
        FileMappingWin32& impl = *std::bit_cast<FileMappingWin32*>(&m_impl);

        // allocate memory to construct a nul terminated, UTF-16 LE, path
        size_t const numChar16     = numBytes_utf16le_from_utf8(_fileName.data(), _fileName.size(), true) >> 1;
        size_t const normNumChar16 = numChar16 + std::max(32ull, numChar16 / 3);
        std::unique_ptr<wchar_t[]> middleBuf = std::make_unique<wchar_t[]>(numChar16);
        std::unique_ptr<wchar_t[]> normBuf   = std::make_unique<wchar_t[]>(normNumChar16);

        uint32_t numCharsWritten        = win32::utf16le_From_utf8(std::bit_cast<char8_t const*>(_fileName.data()),
                                                            _fileName.size(),
                                                            middleBuf.get(),
                                                            numChar16 << 1,
                                                            normBuf.get(),
                                                            normNumChar16 << 1,
                                                            nullptr);
        normBuf[numCharsWritten + 4]    = L'\0';
        DWORD const creationDisposition = _create ? CREATE_NEW : OPEN_EXISTING;
        bool        success = win32::constructPathAndDo({normBuf.get(), static_cast<size_t>(numCharsWritten) << 1},
                                                 [this, creationDisposition](std::wstring_view fullPath) {
            FileMappingWin32& impl = *std::bit_cast<FileMappingWin32*>(&m_impl);
            impl.hFile             = CreateFileW(fullPath.data(),
                                     GENERIC_READ | GENERIC_WRITE,
                                     0,
                                     nullptr,
                                     creationDisposition,
                                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                                     nullptr);
            return impl.hFile != INVALID_HANDLE_VALUE;
        });

        if (!success) // TODO error handling
            win32::errorExit(L"CreateFileW Failed");

        // if the file has zero size, then set it to a multiple of the mapping
        LARGE_INTEGER lFileSz{};
        if (!GetFileSizeEx(impl.hFile, &lFileSz))
            win32::errorExit(L"GetFileSizeEx");

        if (lFileSz.QuadPart < m_chunkSize)
        {
            assert(lFileSz.HighPart == 0);
            uint32_t const offset = m_chunkSize - lFileSz.LowPart;
            if (SetFilePointer(impl.hFile, offset, nullptr, FILE_BEGIN) == INVALID_SET_FILE_POINTER &&
                win32::peekLastError() != NO_ERROR)
                win32::errorExit(L"SetFilePointer to the end");
            if (!SetEndOfFile(impl.hFile))
                win32::errorExit(L"SetEndOfFile failed");
            if (SetFilePointer(impl.hFile, 0, nullptr, FILE_BEGIN) == INVALID_SET_FILE_POINTER &&
                win32::peekLastError() != NO_ERROR)
                win32::errorExit(L"SetFilePointer to beginning failed");
        }

        // A file Mapping does not consume physical memory, it only reserves it https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-file-mapping-object
        // so even if the file is enormous, as long as you don't request too much memory, we are file
        // TODO maybe: Use large/huge pages if SeLockMemoryPrivilege is owned by the current token?
        // TODO Test with a big file to see whether we can map the entire thing and only commit partial views to it (without system page mappings)
        impl.hFileMapping = CreateFileMapping2(impl.hFile, nullptr, FILE_MAP_READ | FILE_MAP_WRITE, PAGE_READWRITE, 0, 0, nullptr, nullptr, 0);
        if (!impl.hFileMapping)
            win32::errorExit(L"CreateFileMapping2 failed");
    }

    __host__ CudaFileMapping::~CudaFileMapping()
    {
        FileMappingWin32& impl = *std::bit_cast<FileMappingWin32*>(&m_impl);
        // TODO Wait for completion?
        if (impl.hFileMapping)
            CloseHandle(impl.hFileMapping);
        if (impl.hFile != INVALID_HANDLE_VALUE)
            CloseHandle(impl.hFile);
    }

    //There is an equilvalent to “threadfence_system()” for the CPU which might help: the “sfence” SSE instruction. (Blocks until all prending writes have been completed).
    //I’m not sure whether you can build a bidirection CPU<->GPU protocol with those instructions… would not be very efficient i guess. (_mm_mfence)
    __device__ void CudaFileMapping::requestChunk(int32_t _chunkIndex, uintptr_t _cudaStream)
    {
        FileMappingWin32&         impl   = *std::bit_cast<FileMappingWin32*>(&m_impl);
        cudaStream_t const        stream = std::bit_cast<cudaStream_t>(_cudaStream);
        cuda::atomic_ref<int32_t> chunkReady{m_chunkReady};
        cuda::atomic_ref<int32_t> chunkIndex{m_chunkRequest};

        int32_t activeMask = __activemask();
        int32_t laneLeader = __ffs(activeMask) - 1;
        if (threadIdx.x % warpSize == laneLeader)
        {
            chunkReady.store(1, cuda::std::memory_order_release);
            chunkIndex.store(_chunkIndex, cuda::std::memory_order_release);
            __threadfence_system();
            //cudaEventRecord(impl.chunkReadyEvent, stream); // TODO remove completely and rely on atomics
        }
        __syncwarp(activeMask);

        // wait for CPU to send the data
        while (chunkReady.load(cuda::std::memory_order_acquire) != 2)
            ; // nanosleep not available for cc < 7.0
    }

    __device__ void CudaFileMapping::signalCompletion()
    {
        int activeMask = __activemask();
        int leaderLane = __ffs(activeMask) - 1;
        if (threadIdx.x % warpSize == leaderLane)
        {
            cuda::atomic_ref<int32_t> chunkRequest{m_chunkRequest};
            chunkRequest.store(0, cuda::std::memory_order_release);
            __threadfence_system();
        }
        __syncwarp(activeMask);
    }

    __host__ bool CudaFileMapping::requestedChunk() const
    {
        std::atomic_ref<int32_t const> chunkReady{m_chunkReady};
        return chunkReady.load(std::memory_order_acquire) == 1;
    }

    // TODO maybe add cudaStream_t?
    __host__ void CudaFileMapping::signalChunkLoaded()
    {
        int32_t           chunkIndex = std::atomic_ref<int32_t>(m_chunkRequest).load(std::memory_order_acquire);
        FileMappingWin32& impl       = *std::bit_cast<FileMappingWin32*>(&m_impl);

        impl.currentView = MapViewOfFile3(impl.hFileMapping,
                                          GetCurrentProcess(),
                                          nullptr,
                                          chunkIndex * m_chunkSize,
                                          m_chunkSize,
                                          0,
                                          PAGE_READWRITE,
                                          nullptr,
                                          0);
        if (impl.currentView)
            win32::errorExit(L"MapViewOfFile3 failed");

        cudaError_t err = cudaMemcpy(target, impl.currentView, m_chunkSize, ::cudaMemcpyHostToDevice);
        if (err != ::cudaSuccess)
            win32::errorExit(win32::quickUtf16leFrom("cudaMemcpy ", cudaGetErrorString(err)).get());

        // make sure all memory transactions are finished (will this work with __managed__ memory too?)
        // if only I could support concurrentManagedAccess, this wouldn't be such a pain...
        _mm_sfence();
    }

    DMT_CPU void CudaFileMapping::waitForCompletion()
    {
        FileMappingWin32& impl = *std::bit_cast<FileMappingWin32*>(&m_impl);

        std::atomic_ref chunkRequest{m_chunkRequest};
        while (chunkRequest.load(std::memory_order_acquire) < 0)
        {
            std::this_thread::yield();
        }

        // copy `chunkSize` from target to mapped portion
        assert(impl.currentView && target);
        cudaError_t err = cudaMemcpy(impl.currentView, target, m_chunkSize, ::cudaMemcpyDeviceToHost);
        if (err != ::cudaSuccess)
            win32::errorExit(win32::quickUtf16leFrom("cudaMemcpy from target to view ", cudaGetErrorString(err)).get());

        std::atomic_ref(m_chunkReady).store(0, std::memory_order_seq_cst);
        _mm_sfence();
    }
} // namespace dmt::os