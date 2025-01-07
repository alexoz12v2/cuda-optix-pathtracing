#pragma once

#include "dmtmacros.h"

#include <bit>
#include <limits>
#include <memory_resource>
#include <type_traits>

#include <cassert>
#include <cstdint>

DMT_MODULE_EXPORT dmt {
    struct MemoryContext;

    using CudaStreamHandle                     = uintptr_t;
    inline constexpr CudaStreamHandle noStream = std::numeric_limits<CudaStreamHandle>::max();
    DMT_CPU inline constexpr bool     isValidHandle(CudaStreamHandle handle) { return handle != noStream; }

    DMT_CPU CudaStreamHandle newStream();
    DMT_CPU void             deleteStream(CudaStreamHandle stream);

    struct CUDAHelloInfo
    {
        size_t  totalMemInBytes;
        int32_t device;
        int32_t warpSize;
        bool    cudaCapable;
    };

    bool logCUDAStatus(MemoryContext * mctx);

    // you need to check support for unified memory allocation with `cudaDevAttrManagedMemory`
    // `cudaMalloc` and `cudaMallocManaged` say this about alignment: "The allocated memory is suitably aligned for any kind of variable"
    DMT_CPU CUDAHelloInfo cudaHello(MemoryContext * mctx);

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
    static constexpr auto is_unified_v = is_unified<T>::value;

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
        ePool = 1 << memoryResouceTypeNumBits,
        // eDevice types
        eCudaMalloc = 2 << memoryResouceTypeNumBits,
        // eAsync types
        eCudaMallocAsync = 3 << memoryResouceTypeNumBits,
        // eUnified type
        eCudaMallocManaged = 4 << memoryResouceTypeNumBits,
    };
    inline constexpr EMemoryResourceType extractCategory(EMemoryResourceType e)
    {
        return static_cast<EMemoryResourceType>(
            static_cast<std::underlying_type_t<EMemoryResourceType>>(e) & memoryResouceTypeMask);
    }
    inline constexpr EMemoryResourceType extractType(EMemoryResourceType e)
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

    class DMT_INTERFACE BaseMemoryResource
    {
    public:
        virtual ~BaseMemoryResource() = default;

        // structures defined inside this CUDA translation unit should dynamic cast to the derived type instead
        virtual void* allocateBytes(size_t sz, size_t align)                                      = 0;
        virtual void  freeBytes(void* ptr, size_t sz, size_t align)                               = 0;
        virtual void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream)       = 0;
        virtual void  freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) = 0;
    };

    enum class EMemoryResourceType : uint32_t;
    inline constexpr size_t alignForMemoryResource([[maybe_unused]] EMemoryResourceType eAlloc)
    {
        return alignof(std::max_align_t); // 8
    }

    size_t sizeForMemoryResouce(EMemoryResourceType type);

    BaseMemoryResource* constructMemoryResourceAt(void* ptr, EMemoryResourceType eAlloc);

    void destroyMemoryResouceAt(BaseMemoryResource * p, EMemoryResourceType eAlloc);

    class UnifiedMemoryResource : public BaseMemoryResource, public std::pmr::memory_resource
    {
    private:
        DMT_CPU void* do_allocate(size_t _Bytes, size_t _Align) override;
        DMT_CPU void  do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override;
        DMT_CPU bool  do_is_equal(memory_resource const& _That) const noexcept override;

        // Inherited via BaseMemoryResource
        void* allocateBytes(size_t sz, size_t align) override;
        void  freeBytes(void* ptr, size_t sz, size_t align) override;
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override;
        void  freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override;
    };

    void switchOnMemoryResoure(EMemoryResourceType eAlloc, BaseMemoryResource * p, size_t * sz, bool destroy);
    EMemoryResourceType categoryOf(BaseMemoryResource * allocator);

    /**
     * Singly linked list whose nodes are blocks of 256 Bytes each, where the last 8 bytes are used for the next pointer
     * and the remaining 248 Bytes to store as many elements as possible, of which we store explicitly copy control functions, destructor
     * insertion should only prepare the memory area and not construct the object. It is the caller responsibility to call construct at if
     * necessary
     * Such list will store a pointer to the BaseMemoryResource and a cuda stream handle at construction. If the handle is valid and the
     * allocator is async, then stream aware allocation is prefered
     */
    class BlockyForwardList
    {
    public:
        static constexpr uint32_t nodeSize  = 256;
        static constexpr uint32_t blockSize = nodeSize - (sizeof(uintptr_t) + sizeof(size_t));
        template <typename T>
        DMT_CPU_GPU explicit BlockyForwardList(BaseMemoryResource* ptr, CudaStreamHandle stream = noStream) :
        stream(stream),
        m_resource(ptr),
        m_elemSize(sizeof(T))
        {
            if (!ptr || m_elemSize > blockSize)
                std::abort();
        }

        // TODO copy control

        DMT_CPU_GPU void lockForRead();
        DMT_CPU_GPU void unlockForRead();
        DMT_CPU_GPU void lockForWrite();
        DMT_CPU_GPU void unlockForWrite();

        CudaStreamHandle stream;

    private:
        struct alignas(std::max_align_t) NodeHeader
        {
            size_t                  occupied;
            void*                   next;
            DMT_CPU_GPU NodeHeader* nextFooter() const
            {
                return std::bit_cast<NodeHeader*>(std::bit_cast<uintptr_t>(next) + blockSize);
            }
        };

    private:
        DMT_CPU_GPU void* allocateNode();
        DMT_CPU_GPU void  freeNode(void* node);

        BaseMemoryResource* m_resource;
        size_t              m_elemSize;
        size_t              m_size = 0;
        NodeHeader*         m_head = nullptr;
        int32_t             m_readCount = 0;
        int32_t             m_writeCount = 0;
    };
} // namespace dmt