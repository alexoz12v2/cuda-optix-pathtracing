#pragma once

#include "dmtmacros.h"
#include <platform/platform-macros.h>

#if !defined(DMT_NEEDS_MODULE)
#include <platform/platform-utils.h>

#include <bit>
#include <limits>
#include <memory_resource>
#include <type_traits>

#include <cassert>
#include <cstdint>
#endif

DMT_MODULE_EXPORT namespace dmt {
    struct MemoryContext;

    using CudaStreamHandle                     = uintptr_t;
    inline constexpr CudaStreamHandle noStream = std::numeric_limits<CudaStreamHandle>::max();
    DMT_CPU inline constexpr bool     isValidHandle(CudaStreamHandle handle) { return handle != noStream; }

    DMT_CPU CudaStreamHandle newStream();
    DMT_CPU void             deleteStream(CudaStreamHandle stream);

    struct CUDAHelloInfo
    {
        struct ScopedLimits
        {
            int32_t maxRegisters;
            int32_t maxSharedMemory;
        };
        struct Extent3D
        {
            int32_t x;
            int32_t y;
            int32_t z;
        };

        size_t       totalMemInBytes;
        int32_t      device;
        int32_t      warpSize;
        ScopedLimits perMultiprocessor;
        int32_t      perMultiprocessorMaxBlocks;
        ScopedLimits perBlock;
        int32_t      perBlockMaxThreads;
        int32_t      multiprocessorCount;
        int32_t      L2CacheBytes;
        int32_t      constantMemoryBytes;
        Extent3D     maxBlockDim;
        Extent3D     maxGridDim;
        uint32_t     cudaCapable           : 1;
        uint32_t     supportsMemoryPools   : 1; // needed for `MemPoolAsyncResource`
        uint32_t     supportsVirtualMemory : 1; // needed for `BuddyMemoryResource`
    };

    bool logCUDAStatus(MemoryContext * mctx);

    // you need to check support for unified memory allocation with `cudaDevAttrManagedMemory`
    // `cudaMalloc` and `cudaMallocManaged` say this about alignment: "The allocated memory is suitably aligned for any kind of variable"
    DMT_CPU CUDAHelloInfo cudaHello(MemoryContext * mctx);

    DMT_GPU int32_t globalThreadIndex();
    DMT_GPU int32_t warpWideThreadIndex();

    // https://committhis.github.io/2020/10/06/cuda-abstractions.html https://gist.github.com/CommitThis/1666517de32893e5dc4c441269f1029a
    // https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_resource/resource.html
    // https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
    // https://stackoverflow.com/questions/27756187/is-extern-c-no-longer-needed-anymore-in-cuda
    // Wrap these functions with extern "C" linkage if you need to export them in a DLL or need to locate them dynamically
    DMT_CPU void*    cudaAllocate(size_t sz);
    DMT_CPU_GPU void cudaDeallocate(void* ptr, size_t sz);

    template <typename T>
    class UnifiedAllocator
    {
    public:
        using value_type            = T;
        using pointer               = value_type*;
        using size_type             = size_t;
        UnifiedAllocator() noexcept = default;
        template <typename U>
        UnifiedAllocator(UnifiedAllocator<U> const&) noexcept {};

        pointer allocate(size_type n, void const* = 0) { return cudaAllocate(n * sizeof(T)); }

        void deallocate(pointer p, size_type n) { cudaDeallocate(p, n * sizeof(T)); }
    };

    template <class T, class U>
    auto operator==(UnifiedAllocator<T> const&, UnifiedAllocator<U> const&)->bool
    {
        return true;
    }

    template <class T, class U>
    auto operator!=(UnifiedAllocator<T> const&, UnifiedAllocator<U> const&)->bool
    {
        return false;
    }

    template <typename T>
    struct is_unified : std::false_type
    {
    };

    template <template <typename, typename> typename Outer, typename Inner>
    struct is_unified<Outer<Inner, UnifiedAllocator<Inner>>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr auto is_unified_v = is_unified<T>::value;

    // TODO include unified memory resouce here
    inline constexpr uint32_t memoryResouceTypeNumBits = 4;
    inline constexpr uint32_t memoryResouceTypeMask    = memoryResouceTypeNumBits - 1;
    enum class EMemoryResourceType : uint32_t
    {
        // first 4 bits dedicated to the allocator category
        eHost    = 0,
        eDevice  = 1,
        eAsync   = 2,
        eUnified = 3,

        // the other 30 bits are for the type
        // eHost types
        eHostToDevMemMap = 5 < memoryResouceTypeNumBits, // host to device
        // eDevice types
        eCudaMalloc = 2 << memoryResouceTypeNumBits,
        // eAsync types
        eCudaMallocAsync = 3 << memoryResouceTypeNumBits,
        eMemPool         = 6 << memoryResouceTypeNumBits,
        // eUnified type
        eCudaMallocManaged = 4 << memoryResouceTypeNumBits,
    };
    DMT_CPU_GPU inline constexpr EMemoryResourceType extractCategory(EMemoryResourceType e)
    {
        return static_cast<EMemoryResourceType>(
            static_cast<std::underlying_type_t<EMemoryResourceType>>(e) & memoryResouceTypeMask);
    }
    DMT_CPU_GPU inline constexpr EMemoryResourceType extractType(EMemoryResourceType e)
    {
        return static_cast<EMemoryResourceType>(
            static_cast<std::underlying_type_t<EMemoryResourceType>>(e) & ~memoryResouceTypeMask);
    }
    inline EMemoryResourceType makeMemResId(EMemoryResourceType category, EMemoryResourceType type)
    {
        using T = std::underlying_type_t<EMemoryResourceType>;
        assert((static_cast<T>(category) & ~memoryResouceTypeMask) == 0 &&
               (static_cast<T>(type) & memoryResouceTypeMask) == 0);
        return static_cast<EMemoryResourceType>(static_cast<T>(category) | static_cast<T>(type));
    }

    class BaseMemoryResource
    {
    public:
        DMT_CPU_GPU void* tryAllocateAsync(size_t sz, size_t align, CudaStreamHandle stream = noStream);
        DMT_CPU_GPU void  tryFreeAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream = noStream);
        DMT_CPU_GPU void* allocateBytes(size_t sz, size_t align);
        DMT_CPU_GPU void  freeBytes(void* ptr, size_t sz, size_t align);
        DMT_CPU void*     allocateBytesAsync(size_t sz, size_t align, CudaStreamHandle stream);
        DMT_CPU void      freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream);
        DMT_CPU_GPU bool  deviceHasAccess(int32_t deviceID) const;
        DMT_CPU_GPU bool  hostHasAccess() const;

        EMemoryResourceType type;

    protected:
        BaseMemoryResource(EMemoryResourceType type) : type(type) {}

    public:
        struct VTableHost
        {
            void* (*allocateBytes)(BaseMemoryResource* pAlloc, size_t sz, size_t align)       = nullptr;
            void (*freeBytes)(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align) = nullptr;
            void* (*allocateBytesAsync)(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream) = nullptr;
            void (*freeBytesAsync)(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream) = nullptr;
            bool (*deviceHasAccess)(BaseMemoryResource const* pAlloc, int32_t deviceID) = nullptr;
            bool (*hostHasAccess)(BaseMemoryResource const* pAlloc)                     = nullptr;
        };
        struct VTableDevice
        {
            void* (*allocateBytes)(BaseMemoryResource* pAlloc, size_t sz, size_t align)       = nullptr;
            void (*freeBytes)(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align) = nullptr;
            bool (*deviceHasAccess)(BaseMemoryResource const* pAlloc, int32_t deviceID)       = nullptr;
            bool (*hostHasAccess)(BaseMemoryResource const* pAlloc)                           = nullptr;
        };
        // you can either store it here or store a pointer
        // storing inline means that every instance of the same derived class duplicates the table
        // but avoids double indirection. Since it is likely that we have 1 instance per allocator type, store it inline
        VTableHost   m_host;
        VTableDevice m_device;
    };

    // Memory Resource Inputs and Types -------------------------------------------------------------------------------

    /**
     * The first allocator you should create in the application, from which all the other allocator and shared data structures
     * (like work queues) should be created. It uses managed memory
     */
    class UnifiedMemoryResource : public BaseMemoryResource, public std::pmr::memory_resource
    {
    public:
        DMT_CPU static UnifiedMemoryResource* create();
        DMT_CPU static void                   destroy(UnifiedMemoryResource* ptr);

    private:
        DMT_CPU UnifiedMemoryResource();

    private:
        DMT_CPU void* do_allocate(size_t _Bytes, size_t _Align) override;
        DMT_CPU void  do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override;
        DMT_CPU bool  do_is_equal(memory_resource const& _That) const noexcept override;

    public:
        static DMT_CPU_GPU void* allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align);
        static DMT_CPU_GPU void  freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align);
        static DMT_CPU void* allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream);
        static DMT_CPU void freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream);
        static DMT_CPU_GPU bool deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID);
        static DMT_CPU_GPU bool hostHasAccess([[maybe_unused]] BaseMemoryResource const* pAlloc);
    };

    struct BuddyResourceSpec
    {
        MemoryContext*             pmctx;
        std::pmr::memory_resource* pHostMemRes;
        size_t                     maxPoolSize; // multiple of 2MB (cc 7.0) (ceil)
        uint32_t minBlockSize; // power of two (ceil). `minBlockSize * minBlocks` should be multiple of 2MB
        uint32_t minBlocks;    // number of device memory blocks to be committed immediately
        int32_t  deviceId;
    };

    struct MemPoolAsyncMemoryResourceSpec
    {
        MemoryContext*             pmctx;
        size_t                     poolSize;         // if 0 OS determined
        size_t                     releaseThreshold; // max memory the pool will retain after a free (eg. max)
        std::pmr::memory_resource* pHostMemRes;
        int32_t                    deviceId;
    };

    // Allocation Utilities for Containers ----------------------------------------------------------------------------

    enum class EMemoryResourceType : uint32_t;
    inline constexpr size_t alignForMemoryResource([[maybe_unused]] EMemoryResourceType eAlloc)
    {
        return alignof(std::max_align_t); // 8
    }

    DMT_CPU_GPU size_t sizeForMemoryResource(EMemoryResourceType type);

    DMT_CPU BaseMemoryResource* constructMemoryResourceAt(void* ptr, EMemoryResourceType eAlloc, void* ctorParam);

    DMT_CPU void destroyMemoryResourceAt(BaseMemoryResource * p, EMemoryResourceType eAlloc);

    DMT_CPU_GPU void
        switchOnMemoryResource(EMemoryResourceType eAlloc, BaseMemoryResource * p, size_t * sz, bool destroy, void* ctorParam);
    DMT_CPU_GPU EMemoryResourceType categoryOf(BaseMemoryResource * allocator);

    // whether allocated memory is accessible from device or host (or both). Independent from the fact the allocation functions
    // can be called by device or host, which is instead encoded in the category
    DMT_CPU_GPU bool isDeviceAllocator(BaseMemoryResource * allocator, int32_t deviceId);
    DMT_CPU_GPU bool isHostAllocator(BaseMemoryResource * allocator);

    struct AllocBundle
    {
        AllocBundle(dmt::UnifiedMemoryResource* unified,
                    dmt::EMemoryResourceType    category,
                    dmt::EMemoryResourceType    type,
                    void*                       ctorParam)
        {
            pUnified         = unified;
            memEnum          = dmt::makeMemResId(category, type);
            memSz            = dmt::sizeForMemoryResource(memEnum);
            memAlign         = dmt::alignForMemoryResource(memEnum);
            pMemBytes        = unified->allocate(memSz, memAlign);
            pMemBytesAligned = dmt::alignTo(pMemBytes, memAlign);
            pMemRes          = dmt::constructMemoryResourceAt(pMemBytesAligned, memEnum, ctorParam);
        }
        AllocBundle(AllocBundle const&)                = delete;
        AllocBundle(AllocBundle&&) noexcept            = delete;
        AllocBundle& operator=(AllocBundle const&)     = delete;
        AllocBundle& operator=(AllocBundle&&) noexcept = delete;
        ~AllocBundle()
        {
            dmt::destroyMemoryResourceAt(pMemRes, memEnum);
            pUnified->deallocate(pMemBytes, memSz, memAlign);
        }

        dmt::UnifiedMemoryResource* pUnified;
        dmt::EMemoryResourceType    memEnum;
        size_t                      memSz;
        size_t                      memAlign;
        void*                       pMemBytes;
        void*                       pMemBytesAligned;
        dmt::BaseMemoryResource*    pMemRes;
    };

    // Containers -----------------------------------------------------------------------------------------------------
    // To share Containers between GPU and CPU, you either need to copy (shallow) them back and forth or place them
    // into

    class BaseDeviceContainer
    {
    public:
        static inline constexpr int32_t leaderWarpIndex = 0;
        DMT_CPU_GPU explicit BaseDeviceContainer(BaseMemoryResource* ptr, CudaStreamHandle stream = noStream) :
        stream(stream),
        m_resource(ptr)
        {
        }

        DMT_CPU_GPU void lockForRead() const;
        DMT_CPU_GPU void unlockForRead() const;
        DMT_CPU_GPU bool lockForWrite() const;
        DMT_CPU_GPU void unlockForWrite() const;
        DMT_CPU_GPU void waitWriter() const;

        CudaStreamHandle stream;

        DMT_CPU_GPU BaseMemoryResource* resource() const noexcept { return m_resource; }

    protected:
        BaseMemoryResource* m_resource;
        mutable int32_t     m_readCount  = 0;
        mutable int32_t     m_writeCount = 0;
    };

    // now works only for std::is_trivial_v<T> types
    class DynaArray : public BaseDeviceContainer
    {
    public:
        DMT_CPU_GPU explicit DynaArray(size_t elemSz, BaseMemoryResource* ptr, CudaStreamHandle stream = noStream) :
        BaseDeviceContainer(ptr, stream),
        m_elemSize(elemSz)
        {
        }

        DMT_CPU_GPU DynaArray(DynaArray const& other);

        DMT_CPU_GPU DynaArray(DynaArray&& other) noexcept;

        DMT_CPU_GPU ~DynaArray() noexcept;

        DMT_CPU_GPU DynaArray& operator=(DynaArray const& other);

        DMT_CPU_GPU DynaArray& operator=(DynaArray&& other) noexcept;

        DMT_CPU_GPU void reserve(size_t newCapacity, bool lock = true);

        DMT_CPU_GPU void clear(bool lock = true) noexcept;

        DMT_CPU_GPU bool push_back(void const* pValue, bool srcHost, bool lock = true);

        DMT_CPU_GPU void pop_back(bool lock = true);

        // assumes you already locked for read
        DMT_CPU_GPU void*       at(size_t index);
        DMT_CPU_GPU void const* atConst(size_t index) const;

        DMT_CPU bool copyToHostSync(void* dest, bool lock = true) const;

        DMT_CPU_GPU size_t size(bool lock = true) const;

        DMT_CPU_GPU size_t capacity(bool lock = true) const;

    private:
        // requires read locked other
        DMT_CPU_GPU void copyFrom(DynaArray const& other);
        DMT_CPU_GPU bool eligibleForAccess(size_t index) const;

        size_t m_elemSize;
        size_t m_capacity = 0;
        size_t m_size     = 0;
        void*  m_head     = nullptr;
    };
} // namespace dmt