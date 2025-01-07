#pragma once

#define DMT_INTERFACE_AS_HEADER
#include "dmtmacros.h"
#include "platform-cuda-utils.h"

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <bit>
#include <memory_resource>

#include <cuda/memory_resource>

namespace dmt {
    template <class MemoryResource>
        requires cuda::mr::resource<MemoryResource>
    void* maybe_allocate_async(MemoryResource& resource, std::size_t size, std::size_t align, cuda::stream_ref stream)
    {
        if constexpr (cuda::mr::async_resource<MemoryResource>)
        {
            return resource.allocate_async(size, align, stream);
        }
        else
        {
            return resource.allocate(size, align);
        }
    }

    DMT_CPU_GPU inline constexpr cuda::stream_ref streamRefFromHandle(CudaStreamHandle handle)
    {
        return {std::bit_cast<::cudaStream_t>(handle)};
    }

    // Memory Resouce Interfaces --------------------------------------------------------------------------------------
    // cannot derive std::pmr::memory_resouce here cause we need __device__ on the allocate
    class DMT_INTERFACE DeviceMemoryReosurce : public BaseMemoryResource
    {
    public:
        // BesaMemoryResouce
        void* allocateBytes(size_t sz, size_t align) override;
        void  freeBytes(void* ptr, size_t sz, size_t align) override;
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override;
        void  freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override;

        DMT_CPU_GPU void* allocate(size_t sz, size_t align);
        DMT_CPU_GPU void  deallocate(void* ptr, size_t sz, size_t align);
        DMT_CPU_GPU bool  operator==(DeviceMemoryReosurce const&) const noexcept;

    private:
        DMT_CPU_GPU virtual void* do_allocate(size_t sz, size_t align)              = 0;
        DMT_CPU_GPU virtual void  do_deallocate(void* ptr, size_t sz, size_t align) = 0;
    };
    static_assert(cuda::mr::resource<DeviceMemoryReosurce>);

    class DMT_INTERFACE CudaAsyncMemoryReosurce : public BaseMemoryResource, public std::pmr::memory_resource
    {
    public:
        void* allocateBytes(size_t sz, size_t align);
        void  freeBytes(void* ptr, size_t sz, size_t align);
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override;
        void  freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override;

        DMT_CPU void* allocate_async(size_t sz, size_t align, cuda::stream_ref stream);
        DMT_CPU void  deallocate_async(void* ptr, size_t sz, size_t align, cuda::stream_ref stream);

    private:
        DMT_CPU virtual void* do_allocate_async(size_t, size_t, cuda::stream_ref)          = 0;
        DMT_CPU virtual void  do_deallocate_async(void*, size_t, size_t, cuda::stream_ref) = 0;
    };
    static_assert(cuda::mr::async_resource<CudaAsyncMemoryReosurce>);

    // Memory Resouce Implementations ---------------------------------------------------------------------------------
    class HostPoolReousce : public DeviceMemoryReosurce, public std::pmr::memory_resource
    {
    private:
        void* allocateBytes(size_t sz, size_t align) override;
        void  freeBytes(void* ptr, size_t sz, size_t align) override;
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override;

        void freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override;

        void* do_allocate(size_t _Bytes, size_t _Align) override;
        void  do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override;
        bool  do_is_equal(memory_resource const& _That) const noexcept override;

        static inline std::pmr::pool_options opts{
            .max_blocks_per_chunk        = 32,
            .largest_required_pool_block = 256,
        };
        // TODO use our multipool allocator
        std::pmr::synchronized_pool_resource m_res{opts};
    };

    class CudaMallocResource : public DeviceMemoryReosurce
    {
    private:
        DMT_CPU_GPU void* do_allocate(size_t sz, [[maybe_unused]] size_t align) override;

        DMT_CPU_GPU void do_deallocate(void* ptr, size_t sz, size_t align) override;
    };

    class CudaMallocAsyncResource : public CudaAsyncMemoryReosurce
    {
    private:
        void*         do_allocate(size_t _Bytes, [[maybe_unused]] size_t _Align) override;
        void          do_deallocate(void* _Ptr, size_t _Bytes, [[maybe_unused]] size_t _Align) override;
        DMT_CPU void* do_allocate_async(size_t sz, [[maybe_unused]] size_t align, cuda::stream_ref stream) override;
        DMT_CPU void  do_deallocate_async(void*                   ptr,
                                          [[maybe_unused]] size_t sz,
                                          [[maybe_unused]] size_t align,
                                          cuda::stream_ref        stream) override;
        bool          do_is_equal(memory_resource const& _That) const noexcept override;
    };

    void* allocateFromCategory(BaseMemoryResource* allocator, size_t sz, size_t align, CudaStreamHandle stream);
    void  freeFromCategory(BaseMemoryResource* allocator, void* ptr, size_t sz, size_t align, CudaStreamHandle stream);
} // namespace dmt
