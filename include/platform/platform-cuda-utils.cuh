#pragma once

#define DMT_INTERFACE_AS_HEADER
#include "dmtmacros.h"
#include "platform/platform-cuda-utils.h"
#include "platform/platform-threadPool.h"

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <bit>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include <cuda.h>
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

    /**
     * Returns the `CUdevice` associated to the primary context created by the CUDA Runtime API
     */
    DMT_CPU CUdevice currentDeviceHandle();

    // Memory Allocation Helpers --------------------------------------------------------------------------------------
    size_t getDeviceAllocationGranularity(int32_t deviceId, CUmemAllocationProp* outProp = nullptr);

    /**
     * Should be called only if the passed device id supports `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
     */
    bool allocateDevicePhysicalMemory(int32_t deviceId, size_t size, CUmemGenericAllocationHandle& out);

    /**
     * Called on some device virtual memory already backed by physical memory, ie `cuMemMap` already called
     * Requires `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`
     */
    void setReadWriteDeviceVirtMemory(int32_t deviceId, CUdeviceptr ptr, size_t size);

    // Memory Resouce Interfaces --------------------------------------------------------------------------------------
    // cannot derive std::pmr::memory_resouce here cause we need __device__ on the allocate
    class DMT_INTERFACE DeviceMemoryReosurce : public BaseMemoryResource
    {
    public:
        friend constexpr void get_property(DeviceMemoryReosurce const&, cuda::mr::device_accessible) noexcept {}
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
    static_assert(cuda::mr::resource<DeviceMemoryReosurce> &&
                  cuda::has_property<DeviceMemoryReosurce, cuda::mr::device_accessible>);

    class DMT_INTERFACE CudaAsyncMemoryReosurce : public BaseMemoryResource, public std::pmr::memory_resource
    {
    public:
        friend constexpr void get_property(CudaAsyncMemoryReosurce const&, cuda::mr::device_accessible) noexcept {}

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
    static_assert(cuda::mr::async_resource<CudaAsyncMemoryReosurce> &&
                  cuda::has_property<CudaAsyncMemoryReosurce, cuda::mr::device_accessible>);

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

    class CudaMallocResource :
    public DeviceMemoryReosurce,
        public cuda::forward_property<CudaMallocResource, DeviceMemoryReosurce>
    {
    private:
        DMT_CPU_GPU void* do_allocate(size_t sz, [[maybe_unused]] size_t align) override;

        DMT_CPU_GPU void do_deallocate(void* ptr, size_t sz, size_t align) override;
    };
    static_assert(cuda::mr::resource<CudaMallocResource> &&
                  cuda::has_property<CudaMallocResource, cuda::mr::device_accessible>);

    class CudaMallocAsyncResource :
    public CudaAsyncMemoryReosurce,
        public cuda::forward_property<CudaMallocAsyncResource, CudaAsyncMemoryReosurce>
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

    class BuddyAsyncResource :
    public CudaAsyncMemoryReosurce,
        public cuda::forward_property<BuddyAsyncResource, CudaAsyncMemoryReosurce>
    {
    public:
        BuddyAsyncResource(BuddyAsyncResourceSpec const& input);
        BuddyAsyncResource(BuddyAsyncResource const& other);
        BuddyAsyncResource(BuddyAsyncResource&& other) noexcept;
        BuddyAsyncResource& operator=(BuddyAsyncResource const& other);
        BuddyAsyncResource& operator=(BuddyAsyncResource&& other) noexcept;
        ~BuddyAsyncResource() override;

    private:
        struct UnifiedControlBlock
        {
            CUdeviceptr ptr;
            size_t      size = 0; // number of chunks physically allocated and mapped, equal to number of handles
            size_t      capacity; // number of handles we can hold in the VLA
            //std::pmr::vector<uint32_t> allocationBitmap;
            std::atomic<int32_t> refCount;
            SpinLock             transactionInFlight;
            // VLA of `CUmemGenericAllocationHandle` here
        };
        static_assert(alignof(UnifiedControlBlock) >= alignof(CUmemGenericAllocationHandle));

    private:
        DMT_CPU void                  cleanup() noexcept;
        CUmemGenericAllocationHandle* vlaStart() const;
        DMT_CPU bool                  grow();

        // Inherited via CudaAsyncMemoryReosurce
        void*         do_allocate(size_t _Bytes, size_t _Align) override;
        void          do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override;
        bool          do_is_equal(memory_resource const& that) const noexcept override;
        DMT_CPU void* do_allocate_async(size_t, size_t, cuda::stream_ref) override;
        DMT_CPU void  do_deallocate_async(void*, size_t, size_t, cuda::stream_ref) override;

    private:
        UnifiedControlBlock* m_ctrlBlock; // when moved from, this is nullptr, and if it is, allocation functions return always nullptr

        // constants which shouldn't change
        size_t m_maxPoolSize;
        size_t m_ctrlBlockReservedVMemBytes;
        size_t m_chunkSize;
        std::shared_mutex m_mtx; // allocation functions use a std::shared_lock on this, while std::lock_guard used by copy control and cleanup
        CUdevice m_device;
        int32_t  m_deviceId;
        uint32_t m_minBlockSize;
    };
} // namespace dmt
