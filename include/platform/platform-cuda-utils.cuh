#pragma once

#define DMT_INTERFACE_AS_HEADER
#include "dmtmacros.h"
#include <platform/platform-macros.h>
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
    class DeviceMemoryReosurce : public BaseMemoryResource
    {
    public:
        friend constexpr void get_property(DeviceMemoryReosurce const&, cuda::mr::device_accessible) noexcept {}

    protected:
        DeviceMemoryReosurce(EMemoryResourceType t, sid_t id) :
        BaseMemoryResource(makeMemResId(EMemoryResourceType::eDevice, t), id)
        {
        }
    };
    static_assert(cuda::has_property<DeviceMemoryReosurce, cuda::mr::device_accessible>);

    class DMT_INTERFACE CudaAsyncMemoryReosurce : public BaseMemoryResource
    {
    protected:
        CudaAsyncMemoryReosurce(EMemoryResourceType t, sid_t sid) :
        BaseMemoryResource(makeMemResId(EMemoryResourceType::eAsync, t), sid)
        {
        }

        friend constexpr void get_property(CudaAsyncMemoryReosurce const&, cuda::mr::device_accessible) noexcept {}
    };
    static_assert(cuda::has_property<CudaAsyncMemoryReosurce, cuda::mr::device_accessible>);

    // Memory Resouce Implementations ---------------------------------------------------------------------------------

    class CudaMallocResource :
    public DeviceMemoryReosurce,
        public cuda::forward_property<CudaMallocResource, DeviceMemoryReosurce>
    {
    public:
        static constexpr sid_t id = "CudaMallocResource"_side;
        DMT_CPU                CudaMallocResource();

        DMT_CPU_GPU void* allocate(size_t sz, [[maybe_unused]] size_t align);
        DMT_CPU_GPU void  deallocate(void* ptr, size_t sz, size_t align);
        DMT_CPU_GPU bool  operator==(CudaMallocResource const&) const noexcept { return true; }

    public:
        static DMT_CPU_GPU void* allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align);
        static DMT_CPU_GPU void  freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align);
        static DMT_CPU void* allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream);
        static DMT_CPU void freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream);
        static inline DMT_CPU_GPU bool deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID)
        {
            return true;
        }
        static inline DMT_CPU_GPU bool hostHasAccess([[maybe_unused]] BaseMemoryResource const* pAlloc)
        {
            return false;
        }
    };
    static_assert(cuda::mr::resource<CudaMallocResource> &&
                  cuda::has_property<CudaMallocResource, cuda::mr::device_accessible>);

    class CudaMallocAsyncResource :
    public CudaAsyncMemoryReosurce,
        public cuda::forward_property<CudaMallocAsyncResource, CudaAsyncMemoryReosurce>
    {
    public:
        static constexpr sid_t id = "CudaMallocAsyncResource"_side;
        DMT_CPU                CudaMallocAsyncResource();

        void*         allocate(size_t _Bytes, [[maybe_unused]] size_t _Align);
        void          deallocate(void* _Ptr, size_t _Bytes, [[maybe_unused]] size_t _Align);
        DMT_CPU void* allocate_async(size_t sz, [[maybe_unused]] size_t align, cuda::stream_ref stream);
        DMT_CPU void deallocate_async(void* ptr, [[maybe_unused]] size_t sz, [[maybe_unused]] size_t align, cuda::stream_ref stream);
        DMT_CPU_GPU bool operator==(CudaMallocAsyncResource const&) const noexcept { return true; }

    public:
        static DMT_CPU_GPU void* allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align);
        static DMT_CPU_GPU void  freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align);
        static DMT_CPU void* allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream);
        static DMT_CPU void freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream);
        static inline DMT_CPU_GPU bool deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID)
        {
            return true;
        }
        static inline DMT_CPU_GPU bool hostHasAccess([[maybe_unused]] BaseMemoryResource const* pAlloc)
        {
            return false;
        }
    };
    static_assert(cuda::mr::async_resource<CudaMallocAsyncResource> &&
                  cuda::has_property<CudaMallocAsyncResource, cuda::mr::device_accessible>);

    class BuddyMemoryResource : public BaseMemoryResource
    {
        friend constexpr void get_property(BuddyMemoryResource const&, cuda::mr::device_accessible) noexcept {}

    public:
        static constexpr sid_t       id = "BuddyMemoryResource"_side;
        DMT_CPU                      BuddyMemoryResource(BuddyResourceSpec const& input);
        DMT_CPU                      BuddyMemoryResource(BuddyMemoryResource const& other);
        DMT_CPU                      BuddyMemoryResource(BuddyMemoryResource&& other) noexcept;
        DMT_CPU BuddyMemoryResource& operator=(BuddyMemoryResource const& other);
        DMT_CPU BuddyMemoryResource& operator=(BuddyMemoryResource&& other) noexcept;
        DMT_CPU ~BuddyMemoryResource();

        DMT_CPU_GPU size_t maxPoolSize() const noexcept { return m_maxPoolSize; }
        DMT_CPU_GPU size_t maxBlockSize() const noexcept { return m_chunkSize; }

        // Inherited via std::pmr::memory_resource
        DMT_CPU void*    allocate(size_t _Bytes, size_t _Align);
        DMT_CPU void     deallocate(void* _Ptr, size_t _Bytes, size_t _Align);
        DMT_CPU_GPU bool operator==(BuddyMemoryResource const& that) const noexcept;

    public:
        static DMT_CPU_GPU void* allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align);
        static DMT_CPU_GPU void  freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align);
        static inline DMT_CPU void* allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream)
        {
            assert(false);
            return nullptr;
        }
        static inline DMT_CPU void freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
        {
            assert(false);
        }
        static DMT_CPU_GPU bool        deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID);
        static inline DMT_CPU_GPU bool hostHasAccess([[maybe_unused]] BaseMemoryResource const* pAlloc)
        {
            return false;
        }

    private:
        enum EBitTree : uint8_t
        {
            eFree = 0,
            eAllocated,
            eHasChildren,
            eInvalid
        };
        using BitTree = std::pmr::vector<EBitTree>;
        struct UnifiedControlBlock
        {
            UnifiedControlBlock(std::pmr::memory_resource* res) : allocationBitmap(res) {}

            CUdeviceptr ptr      = 0;
            size_t      size     = 0; // number of chunks physically allocated and mapped, equal to number of handles
            size_t      capacity = 0; // number of handles we can hold in the VLA
            std::pmr::vector<EBitTree>   allocationBitmap;
            mutable std::atomic<int32_t> refCount;
            mutable SpinLock             transactionInFlight;
            // VLA of `CUmemGenericAllocationHandle` here
        };
        static_assert(alignof(UnifiedControlBlock) >= alignof(CUmemGenericAllocationHandle));

    private:
        DMT_CPU void                  cleanup() noexcept;
        CUmemGenericAllocationHandle* vlaStart() const;
        DMT_CPU bool                  grow();
        DMT_CPU size_t                minOrder() const;
        DMT_CPU size_t                blockToOrder(size_t size) const;
        DMT_CPU size_t                alignUpToBlock(size_t size) const;
        DMT_CPU void                  split(size_t order, size_t nodeIndex);
        DMT_CPU bool                  coalesce(size_t parentIndex, size_t parentLevel);

    private:
        UnifiedControlBlock* m_ctrlBlock; // when moved from, this is nullptr, and if it is, allocation functions return always nullptr

        // constants which shouldn't change
        size_t m_maxPoolSize;
        size_t m_ctrlBlockReservedVMemBytes;
        size_t m_chunkSize;
        mutable std::shared_mutex m_mtx; // allocation functions use a std::shared_lock on this, while std::lock_guard used by copy control and cleanup
        CUdevice m_deviceHnd;
        int32_t  m_deviceId;
        uint32_t m_minBlockSize;
    };
    static_assert(cuda::mr::resource<BuddyMemoryResource> &&
                  cuda::has_property<BuddyMemoryResource, cuda::mr::device_accessible>);

    // Reference for Stream Ordered Allocation (Runtime API): https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator-intro
    class MemPoolAsyncMemoryResource :
    public CudaAsyncMemoryReosurce,
        public cuda::forward_property<MemPoolAsyncMemoryResource, CudaAsyncMemoryReosurce>
    {
    public:
        static constexpr sid_t              id = "MemPoolAsyncMemoryResource"_side;
        DMT_CPU                             MemPoolAsyncMemoryResource(MemPoolAsyncMemoryResourceSpec const& input);
        DMT_CPU                             MemPoolAsyncMemoryResource(MemPoolAsyncMemoryResource const& other);
        DMT_CPU                             MemPoolAsyncMemoryResource(MemPoolAsyncMemoryResource&& other) noexcept;
        DMT_CPU MemPoolAsyncMemoryResource& operator=(MemPoolAsyncMemoryResource const& other);
        DMT_CPU MemPoolAsyncMemoryResource& operator=(MemPoolAsyncMemoryResource&& other) noexcept;
        DMT_CPU ~MemPoolAsyncMemoryResource() noexcept;

        DMT_CPU_GPU size_t poolSize() const noexcept { return m_poolSize; }

        DMT_CPU void* allocate(size_t _Bytes, size_t _Align);
        DMT_CPU void  deallocate(void* _Ptr, size_t _Bytes, size_t _Align);
        DMT_CPU void* allocate_async(size_t, size_t, cuda::stream_ref);
        DMT_CPU void  deallocate_async(void*, size_t, size_t, cuda::stream_ref);

        DMT_CPU_GPU bool operator==(MemPoolAsyncMemoryResource const& _That) const noexcept;

    private:
        DMT_CPU void  cleanup() noexcept;
        DMT_CPU void* performAlloc(CUstream streamRef, size_t sz);

    public:
        static DMT_CPU_GPU void* allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align);
        static DMT_CPU_GPU void  freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align);
        static DMT_CPU void* allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream);
        static DMT_CPU void freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream);
        static DMT_CPU_GPU bool        deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID);
        static inline DMT_CPU_GPU bool hostHasAccess([[maybe_unused]] BaseMemoryResource const* pAlloc)
        {
            return false;
        }

    private:
        struct ControlBlock
        {
            CUmemoryPool                 memPool;
            CUstream                     defaultStream;
            mutable std::atomic<int32_t> refCount;
            mutable SpinLock             transactionInFlight;
        };

    private:
        mutable std::shared_mutex  m_mtx; // shared_lock on allocation, lock_guard on copy control
        ControlBlock*              m_ctrlBlock;
        std::pmr::memory_resource* m_hostCtrlRes;
        size_t                     m_poolSize;
        int32_t                    m_deviceId;
    };
    static_assert(cuda::mr::async_resource<MemPoolAsyncMemoryResource> &&
                  cuda::has_property<MemPoolAsyncMemoryResource, cuda::mr::device_accessible>);
} // namespace dmt
