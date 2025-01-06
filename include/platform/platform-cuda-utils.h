#pragma once

#include "dmtmacros.h"

#include <memory_resource>
#include <type_traits>

DMT_MODULE_EXPORT dmt {
    struct MemoryContext;

    struct CUDAHelloInfo
    {
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

    class UnifiedMemoryResource : public std::pmr::memory_resource
    {
    private:
        DMT_CPU void* do_allocate(size_t _Bytes, size_t _Align) override;
        DMT_CPU void  do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override;
        DMT_CPU bool  do_is_equal(memory_resource const& _That) const noexcept override;
    };

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
} // namespace dmt