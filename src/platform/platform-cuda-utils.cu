#include "platform-cuda-utils.cuh"

#define DMT_INTERFACE_AS_HEADER
#include "platform-cuda-utils.h"
#include "platform-memory.h"

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <bit>

#include <cstdint>
#include <cuda.h>
#include <cuda/memory_resource>

namespace dmt {
    bool logCUDAStatus(MemoryContext* mctx)
    {
        if (cudaError_t err = cudaPeekAtLastError(); err != ::cudaSuccess)
        {
            mctx->pctx.error("CUDA error: {}", {cudaGetErrorString(err)});
            return false;
        }
        return true;
    }

    DMT_CPU CudaStreamHandle newStream()
    {
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) == ::cudaSuccess)
            return std::bit_cast<CudaStreamHandle>(stream);
        else
            return noStream;
    }

    DMT_CPU CUdevice currentDeviceHandle()
    {
        CUdevice ret;
        int32_t  deviceId = 0;
        assert(cudaGetDevice(&deviceId) == ::cudaSuccess);
        assert(cuDeviceGet(&ret, deviceId) == ::CUDA_SUCCESS);
        return ret;
    }

    DMT_CPU void deleteStream(CudaStreamHandle stream)
    {
        if (stream != noStream && stream != 0)
        {
            cudaStream_t cStream = std::bit_cast<cudaStream_t>(stream);
            assert(cudaStreamDestroy(cStream) == ::cudaSuccess);
        }
    }

    DMT_CPU CUDAHelloInfo cudaHello(MemoryContext* mctx)
    {
        CUDAHelloInfo ret;
        ret.cudaCapable = false;
        // Force cuda context lazy initialization (driver and runtime interop:
        // https://stackoverflow.com/questions/60132426/how-can-i-mix-cuda-driver-api-with-cuda-runtime-api
        int32_t count = 0;
        if (cudaGetDeviceCount(&count) != ::cudaSuccess || count <= 0)
        {
            mctx->pctx.error("Couldn't find any suitable CUDA capable devices in the current system");
            return ret;
        }

        int32_t        device = -1;
        cudaDeviceProp desiredProps{};
        desiredProps.major             = 6; // minimum compute capability = 6.0
        desiredProps.canMapHostMemory  = 1;
        desiredProps.managedMemory     = 1;
        desiredProps.concurrentKernels = 1;

        if (cudaChooseDevice(&device, &desiredProps) != ::cudaSuccess)
        {
            mctx->pctx.error("Couldn't find any CUDA device which suits the desired requiresments");
            return ret;
        }
        ret.cudaCapable = true;
        ret.device      = device;

        cudaDeviceProp actualProps{};
        if (cudaGetDeviceProperties(&actualProps, device) != ::cudaSuccess)
        {
            mctx->pctx.error("Couldn't get device {} properties", {device});
            return ret;
        }
        mctx->pctx.log("Chosed Device: {} ({})", {std::string_view{actualProps.name}, device});
        mctx->pctx.log("Compute Capability: {}.{}", {actualProps.major, actualProps.minor});
        assert(actualProps.managedMemory && actualProps.canMapHostMemory);

        ret.warpSize = actualProps.warpSize;

        // forrce CUDA context initialization (after this, you can use the CUDA driver API)
        // the context, if needed, can be fetched with `cuCtxGetCurrent`
        if (cudaFree(nullptr) != ::cudaSuccess)
        {
            mctx->pctx.error("Couldn't initialize CUDA context");
            ret.cudaCapable = false;
            return ret;
        }
        size_t totalBytes = ret.totalMemInBytes = 0;
        if (cudaMemGetInfo(nullptr, &totalBytes) != ::cudaSuccess)
            mctx->pctx.error("Couldn't get the total Memory in bytes of the device");
        else
        {
            ret.totalMemInBytes = totalBytes;
            mctx->pctx.log("Total Device Memory: {}", {ret.totalMemInBytes});
        }

        if (actualProps.canMapHostMemory)
        { // all flags starts with `cudaDevice*`
            if (cudaSetDeviceFlags(cudaDeviceMapHost) != ::cudaSuccess)
            {
                mctx->pctx.error("Failed to enable device flags for pin map host memory");
            }
        }

        // check current device support for for CU_DEVICE_ATTRIBUTE_MEMORY_POOLS and CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY
        CUdevice deviceHandle;
        assert(cuDeviceGet(&deviceHandle, device) == ::CUDA_SUCCESS);
        int32_t support = 0;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, deviceHandle) == ::CUDA_SUCCESS);
        assert(support <= 1);
        ret.supportsMemoryPools = support;

        // support for `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` and related
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, deviceHandle) ==
               ::CUDA_SUCCESS);
        assert(support <= 1);
        ret.supportsVirtualMemory = support;

        // various inofration
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, deviceHandle) ==
               ::CUDA_SUCCESS);
        ret.perMultiprocessorMaxBlocks = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, deviceHandle) ==
               ::CUDA_SUCCESS);
        ret.perMultiprocessor.maxRegisters = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, deviceHandle) ==
               ::CUDA_SUCCESS);
        ret.perMultiprocessor.maxSharedMemory = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, deviceHandle) == ::CUDA_SUCCESS);
        ret.L2CacheBytes = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceHandle) == ::CUDA_SUCCESS);
        ret.multiprocessorCount = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, deviceHandle) == ::CUDA_SUCCESS);
        ret.perBlock.maxRegisters = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, deviceHandle) ==
               ::CUDA_SUCCESS);
        ret.perBlock.maxSharedMemory = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, deviceHandle) == ::CUDA_SUCCESS);
        ret.constantMemoryBytes = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, deviceHandle) == ::CUDA_SUCCESS);
        ret.perBlockMaxThreads = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, deviceHandle) == ::CUDA_SUCCESS);
        ret.maxBlockDim.x = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, deviceHandle) == ::CUDA_SUCCESS);
        ret.maxBlockDim.y = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, deviceHandle) == ::CUDA_SUCCESS);
        ret.maxBlockDim.z = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, deviceHandle) == ::CUDA_SUCCESS);
        ret.maxGridDim.x = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, deviceHandle) == ::CUDA_SUCCESS);
        ret.maxGridDim.y = support;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, deviceHandle) == ::CUDA_SUCCESS);
        ret.maxGridDim.z = support;

        // More details about interop between driver and runtime: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html
        // what concerns us is that
        // `CUStream` and `cudaStream_t`
        // `CUevent`  and `cudaEvent_t`
        // `CUarray`  and `cudaArray_t`
        // `CUgraphicsResource` and `cudaGraphicsResource_t`
        // `CUtexObject` and `cudaTextureObject_t`
        // `CUSurfObject` and `cudaSurfaceObject_t`
        // `CUfunction` and `cudaFunction_t`
        // are *all interchangeable* by `static_cast`

        return ret;
    }

    DMT_CPU void* cudaAllocate(size_t sz)
    {
        void*       tmp = nullptr;
        cudaError_t err = cudaMallocManaged(&tmp, sz);
        if (err != ::cudaSuccess)
            return nullptr;
        return tmp;
    }

    DMT_CPU_GPU void cudaDeallocate(void* ptr, size_t sz)
    {
        if (ptr)
            cudaFree(ptr);
    }

    DMT_CPU void* UnifiedMemoryResource::do_allocate(size_t _Bytes, size_t _Align) { return cudaAllocate(_Bytes); }

    DMT_CPU void UnifiedMemoryResource::do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
        cudaDeallocate(_Ptr, _Bytes);
    }

    DMT_CPU bool UnifiedMemoryResource::do_is_equal(memory_resource const& _That) const noexcept { return true; }

    void* UnifiedMemoryResource::allocateBytes(size_t sz, size_t align) { return do_allocate(sz, align); }

    void UnifiedMemoryResource::freeBytes(void* ptr, size_t sz, size_t align) { return do_deallocate(ptr, sz, align); }

    void* UnifiedMemoryResource::allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
        return nullptr;
    }

    void UnifiedMemoryResource::freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
    }

    // ----------------------------------------------------------------------------------------------------------------

    // Memory Resouce Interfaces --------------------------------------------------------------------------------------
    // cannot derive std::pmr::memory_resouce here cause we need __device__ on the allocate
    void* DeviceMemoryReosurce::allocateBytes(size_t sz, size_t align) { return do_allocate(sz, align); }
    void  DeviceMemoryReosurce::freeBytes(void* ptr, size_t sz, size_t align) { do_deallocate(ptr, sz, align); }
    void* DeviceMemoryReosurce::allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
        return nullptr;
    }
    void DeviceMemoryReosurce::freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
    }

    DMT_CPU_GPU void* DeviceMemoryReosurce::allocate(size_t sz, size_t align) { return do_allocate(sz, align); }
    DMT_CPU_GPU void  DeviceMemoryReosurce::deallocate(void* ptr, size_t sz, size_t align)
    {
        do_deallocate(ptr, sz, align);
    }
    DMT_CPU_GPU bool DeviceMemoryReosurce::operator==(DeviceMemoryReosurce const&) const noexcept { return true; }

    void* CudaAsyncMemoryReosurce::allocateBytes(size_t sz, size_t align) { return allocate(sz, align); }
    void  CudaAsyncMemoryReosurce::freeBytes(void* ptr, size_t sz, size_t align) { return deallocate(ptr, sz, align); }
    void* CudaAsyncMemoryReosurce::allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(isValidHandle(stream));
        return do_allocate_async(sz, align, streamRefFromHandle(stream));
    }
    void CudaAsyncMemoryReosurce::freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(isValidHandle(stream));
        do_deallocate_async(ptr, sz, align, streamRefFromHandle(stream));
    }

    DMT_CPU void* CudaAsyncMemoryReosurce::allocate_async(size_t sz, size_t align, cuda::stream_ref stream)
    {
        assert((align & (align - 1)) == 0 && "alignment should be a power of two");
        return do_allocate_async(sz, align, stream);
    }
    DMT_CPU void CudaAsyncMemoryReosurce::deallocate_async(void* ptr, size_t sz, size_t align, cuda::stream_ref stream)
    {
        assert((align & (align - 1)) == 0 && "alignment should be a power of two");
        do_deallocate_async(ptr, sz, align, stream);
    }

    // Memory Allocation Helpers --------------------------------------------------------------------------------------
    size_t getDeviceAllocationGranularity(int32_t deviceId, CUmemAllocationProp* outProp)
    {
        CUmemAllocationProp prop = {};
        prop.type                = ::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = ::CU_MEM_LOCATION_TYPE_DEVICE; // allocate memory on device whose id is given
        prop.location.id         = deviceId;
        size_t granularity       = 0;
        assert(cuMemGetAllocationGranularity(&granularity, &prop, ::CU_MEM_ALLOC_GRANULARITY_MINIMUM) == ::CUDA_SUCCESS);
        if (outProp)
            *outProp = prop;
        return granularity;
    }

    bool allocateDevicePhysicalMemory(int32_t deviceId, size_t size, CUmemGenericAllocationHandle& out)
    {
        CUmemAllocationProp prop        = {};
        size_t              granularity = getDeviceAllocationGranularity(deviceId, &prop);
        size                            = roundUpToNextMultipleOf(size, granularity);
        CUmemGenericAllocationHandle handle;
        auto                         result = cuMemCreate(&handle, size, &prop, 0);
        if (result == ::CUDA_SUCCESS)
        {
            out = handle;
            return true;
        }
        else if (result == ::CUDA_ERROR_OUT_OF_MEMORY)
            return false;
        else
        {
            assert(false);
            return false;
        }
    }

    void setReadWriteDeviceVirtMemory(int32_t deviceId, CUdeviceptr ptr, size_t size)
    {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type   = ::CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id     = deviceId;
        accessDesc.flags           = ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        assert(cuMemSetAccess(ptr, size, &accessDesc, 1) == ::CUDA_SUCCESS);
    }

    // Memory Resouce Implementations ---------------------------------------------------------------------------------
    void* HostPoolResource::do_allocate(size_t _Bytes, size_t _Align) { return m_res.allocate(_Bytes, _Align); }
    void  HostPoolResource::do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
        m_res.deallocate(_Ptr, _Bytes, _Align);
    }
    bool HostPoolResource::do_is_equal(memory_resource const& _That) const noexcept { return m_res == _That; }

    DMT_CPU_GPU void* CudaMallocResource::do_allocate(size_t sz, [[maybe_unused]] size_t align)
    {
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, sz) != ::cudaSuccess)
            return nullptr;
        return tmp;
    }

    DMT_CPU_GPU void CudaMallocResource::do_deallocate(void* ptr, size_t sz, size_t align)
    {
        assert(cudaFree(ptr) == ::cudaSuccess);
    }

    void* CudaMallocAsyncResource::do_allocate(size_t _Bytes, [[maybe_unused]] size_t _Align)
    {
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, _Bytes) != ::cudaSuccess)
            return nullptr;
        return tmp;
    }
    void CudaMallocAsyncResource::do_deallocate(void* _Ptr, size_t _Bytes, [[maybe_unused]] size_t _Align)
    {
        assert(cudaFree(_Ptr) == ::cudaSuccess);
    }
    DMT_CPU void* CudaMallocAsyncResource::do_allocate_async(size_t sz, [[maybe_unused]] size_t align, cuda::stream_ref stream)
    {
        void* tmp = nullptr;
        if (cudaMallocAsync(&tmp, sz, stream.get()) != ::cudaSuccess)
            return nullptr;
        return tmp;
    }
    DMT_CPU void CudaMallocAsyncResource::do_deallocate_async(
        void*                   ptr,
        [[maybe_unused]] size_t sz,
        [[maybe_unused]] size_t align,
        cuda::stream_ref        stream)
    {
        assert(cudaFreeAsync(ptr, stream.get()) == ::cudaSuccess);
    }
    bool CudaMallocAsyncResource::do_is_equal(memory_resource const& _That) const noexcept { return true; }


    BuddyMemoryResource::BuddyMemoryResource(BuddyResourceSpec const& input) :
    m_deviceId(input.deviceId),
    m_minBlockSize(static_cast<uint32_t>(nextPOT(input.minBlockSize)))
    { // TODO better? allocation functions without class? global context map {pid, idx -> ctx}
#if defined(__CUDA_ARCH__)
        // you shouldn't be here
        m_ctrlBlock = nullptr;
#else
        assert(input.pHostMemRes);
        assert(cuDeviceGet(&m_device, m_deviceId) == ::CUDA_SUCCESS);
        int32_t support = 0;
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, m_device) == ::CUDA_SUCCESS);
        if (support == 0)
        {
            input.pmctx->pctx.error("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED not supported on the given device");
            std::abort();
        }
        assert(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, m_device) ==
               ::CUDA_SUCCESS);
        if (support == 0)
        {
            input.pmctx->pctx.error(
                "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED not supported on given device");
            std::abort();
        }

        // compute, from max pool size, the amount of virtual address space to reserve on the host for the Control block
        size_t const granularity     = getDeviceAllocationGranularity(m_deviceId);
        m_maxPoolSize                = roundUpToNextMultipleOf(input.maxPoolSize, granularity);
        m_ctrlBlockReservedVMemBytes = sizeof(UnifiedControlBlock) +
                                       ceilDiv(m_maxPoolSize, sizeof(CUmemGenericAllocationHandle));
        void* reservedHostSpace = reserveVirtualAddressSpace(m_ctrlBlockReservedVMemBytes);
        if (!reservedHostSpace)
        {
            input.pmctx->pctx.error("Couldn't reserve host virtual address space for {} Bytes of metadata",
                                    {m_ctrlBlockReservedVMemBytes});
            std::abort();
        }
        // commit first page, construct control block, compute initial capacity
        if (!commitPhysicalMemory(reservedHostSpace, toUnderlying(EPageSize::e4KB)))
        {
            input.pmctx->pctx.error("Couldn't commit first 4KB physical memory page for allocater metadata");
            std::abort();
        }
        size_t initialCapacity = toUnderlying(EPageSize::e4KB) - sizeof(UnifiedControlBlock);
        assert(initialCapacity % sizeof(CUmemGenericAllocationHandle) == 0);
        initialCapacity /= sizeof(CUmemGenericAllocationHandle);

        m_ctrlBlock = std::construct_at(std::bit_cast<UnifiedControlBlock*>(reservedHostSpace), input.pHostMemRes);
        m_ctrlBlock->refCount.store(1, std::memory_order_seq_cst);
        m_ctrlBlock->capacity = initialCapacity;

        // compute number of initial handles
        m_chunkSize = roundUpToNextMultipleOf(input.minBlocks * static_cast<size_t>(nextPOT(input.minBlockSize)), granularity);
        assert((m_chunkSize < m_maxPoolSize) && (m_maxPoolSize % m_chunkSize == 0));

        // allocate physical memory for first device memory chunk, and reserve virtual address space (has to be paired with `cuMemAddressFree`)
        assert(cuMemAddressReserve(&m_ctrlBlock->ptr, m_maxPoolSize, 0 /*natural align*/, 0 /*hint address*/, 0 /*flags*/) ==
               ::CUDA_SUCCESS);
        CUmemGenericAllocationHandle* arr = vlaStart();
        if (!allocateDevicePhysicalMemory(m_deviceId, m_chunkSize, arr[0]))
        {
            input.pmctx->pctx.error("Couldn't allocate first Device memory chunk, device memory already exhausted");
            std::abort();
        }
        ++m_ctrlBlock->size;

        // map the first device memory chunk
        assert(cuMemMap(m_ctrlBlock->ptr, m_chunkSize, 0, arr[0], 0) == ::CUDA_SUCCESS);
        setReadWriteDeviceVirtMemory(m_deviceId, m_ctrlBlock->ptr, m_chunkSize);

        // allocate the metadata all in one shot, and memset it to
        assert(m_maxPoolSize % m_minBlockSize == 0);
        size_t const completeBinaryTreeNodes = (m_maxPoolSize / m_minBlockSize) << 1;
        m_ctrlBlock->allocationBitmap.resize(completeBinaryTreeNodes);
        std::ranges::fill(m_ctrlBlock->allocationBitmap, eInvalid);

        // this isn't a "pure" buddy system. The allocation actually has a maximum size, hence subdivide the
        // tree until you get nodes representing blocks of `m_chunkSize`
        size_t const minOrderChunk = minOrder();
        for (size_t level = 0; level < minOrderChunk; ++level)
        {
            size_t numNodes = 1ULL << level;
            for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx)
            {
                size_t const index                   = ((1ULL << level) - 1) + nodeIdx; // Calculate index
                m_ctrlBlock->allocationBitmap[index] = eHasChildren;
            }
        }
        for (size_t nodeIdx = 0; nodeIdx < (1ULL << minOrderChunk); ++nodeIdx)
        {
            size_t const index                   = ((1ULL << minOrderChunk) - 1) + nodeIdx; // Calculate index
            m_ctrlBlock->allocationBitmap[index] = eFree;
        }
#endif
    }

    BuddyMemoryResource::BuddyMemoryResource(BuddyMemoryResource const& other)
    {
        // shouldn't be necessary to lock `transactionInFlight`, cause no device allocation is needed here
        std::shared_lock slk{other.m_mtx};
        m_ctrlBlock                  = other.m_ctrlBlock;
        m_maxPoolSize                = other.m_maxPoolSize;
        m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
        m_chunkSize                  = other.m_chunkSize;
        m_device                     = other.m_device;
        m_deviceId                   = other.m_deviceId;
        m_minBlockSize               = other.m_minBlockSize;

        std::lock_guard lk{m_ctrlBlock->transactionInFlight};
        m_ctrlBlock->refCount.fetch_add(1, std::memory_order_seq_cst);
    }

    BuddyMemoryResource::BuddyMemoryResource(BuddyMemoryResource&& other) noexcept
    {
        // shouldn't be necessary to lock `transactionInFlight`, cause no device allocation is needed here
        std::lock_guard lk{other.m_mtx};
        m_ctrlBlock                  = std::exchange(other.m_ctrlBlock, nullptr);
        m_maxPoolSize                = other.m_maxPoolSize;
        m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
        m_chunkSize                  = other.m_chunkSize;
        m_device                     = other.m_device;
        m_deviceId                   = other.m_deviceId;
        m_minBlockSize               = other.m_minBlockSize;
    }

    BuddyMemoryResource& BuddyMemoryResource::operator=(BuddyMemoryResource const& other)
    {
        if (this != &other)
        {
            cleanup();
            std::lock_guard  lk{m_mtx};
            std::shared_lock slk{other.m_mtx};

            m_ctrlBlock                  = other.m_ctrlBlock;
            m_maxPoolSize                = other.m_maxPoolSize;
            m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
            m_chunkSize                  = other.m_chunkSize;
            m_device                     = other.m_device;
            m_deviceId                   = other.m_deviceId;
            m_minBlockSize               = other.m_minBlockSize;
            m_ctrlBlock->refCount.fetch_add(1, std::memory_order_seq_cst);
            // shouldn't be necessary to lock `transactionInFlight`, cause no device allocation is needed here
        }
        return *this;
    }

    BuddyMemoryResource& BuddyMemoryResource::operator=(BuddyMemoryResource&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            std::lock_guard lk{m_mtx};
            std::lock_guard olk{other.m_mtx};

            m_ctrlBlock                  = std::exchange(other.m_ctrlBlock, nullptr);
            m_maxPoolSize                = other.m_maxPoolSize;
            m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
            m_chunkSize                  = other.m_chunkSize;
            m_device                     = other.m_device;
            m_deviceId                   = other.m_deviceId;
            m_minBlockSize               = other.m_minBlockSize;
            // shouldn't be necessary to lock `transactionInFlight`, cause no device allocation is needed here
        }
        return *this;
    }

    DMT_CPU void BuddyMemoryResource::cleanup() noexcept
    {
        if (!m_ctrlBlock)
            return;

        { // shared lock scope
            std::shared_lock slk{m_mtx};
            m_ctrlBlock->transactionInFlight.lock();
            if (m_ctrlBlock->refCount.fetch_sub(1, std::memory_order_seq_cst) > 0)
            {
                m_ctrlBlock->transactionInFlight.unlock();
                return;
            }
        }
        std::lock_guard lk{m_mtx};
        if (!m_ctrlBlock)
            return;

        // unmap and deallocate all device memory chunks
        CUmemGenericAllocationHandle* arr = vlaStart();
        CUdeviceptr                   ptr = m_ctrlBlock->ptr;
        for (size_t i = 0; i < m_ctrlBlock->size; ++i)
        {
            assert(cuMemUnmap(ptr, m_chunkSize) == ::CUDA_SUCCESS);
            assert(cuMemRelease(arr[i]) == ::CUDA_SUCCESS);
            ptr += m_chunkSize;
        }

        // free device virtual address reservation
        assert(cuMemAddressFree(m_ctrlBlock->ptr, m_maxPoolSize) == ::CUDA_SUCCESS);

        std::destroy_at(m_ctrlBlock);

        // compute number of host committed pages, decommit them
        size_t pageCount = m_ctrlBlock->capacity * sizeof(CUmemGenericAllocationHandle) + sizeof(UnifiedControlBlock);
        size_t const pageSz = toUnderlying(EPageSize::e4KB);
        assert(pageCount % pageSz == 0);
        pageCount /= pageSz;
        uintptr_t address = std::bit_cast<uintptr_t>(m_ctrlBlock);
        for (size_t i = 0; i < pageCount; ++i)
        { // first iteration will deallocate the lock, so no need to unlock it
            decommitPage(std::bit_cast<void*>(address), pageSz);
            address += pageSz;
        }

        // free host virtual address space reservation
        freeVirtualAddressSpace(std::bit_cast<void*>(m_ctrlBlock), m_ctrlBlockReservedVMemBytes);
        m_ctrlBlock = nullptr;
    }

    CUmemGenericAllocationHandle* BuddyMemoryResource::vlaStart() const
    {
        assert(m_ctrlBlock);
        return std::bit_cast<CUmemGenericAllocationHandle*>(
            std::bit_cast<uintptr_t>(m_ctrlBlock) + sizeof(UnifiedControlBlock));
    }

    DMT_CPU bool BuddyMemoryResource::grow()
    { // assume you already own the spinlock
        assert(m_ctrlBlock);

        // check if capacity is enough
        if (m_ctrlBlock->capacity <= m_ctrlBlock->size)
        {
            void*  address = std::bit_cast<void*>(std::bit_cast<uintptr_t>(m_ctrlBlock) + sizeof(UnifiedControlBlock) +
                                                 sizeof(CUmemGenericAllocationHandle) * m_ctrlBlock->capacity);
            size_t pageSz  = toUnderlying(EPageSize::e4KB);
            assert(pageSz % sizeof(CUmemGenericAllocationHandle) == 0);
            if (!commitPhysicalMemory(address, pageSz))
                return false;

            m_ctrlBlock->capacity += pageSz / sizeof(CUmemGenericAllocationHandle);
        }

        // allocate and map next chunk of device memory
        size_t const                  offset = m_chunkSize * m_ctrlBlock->size;
        CUmemGenericAllocationHandle* arr    = vlaStart();
        if (!allocateDevicePhysicalMemory(m_deviceId, m_chunkSize, arr[m_ctrlBlock->size]))
            return false;

        assert(cuMemMap(m_ctrlBlock->ptr + offset, m_chunkSize, 0, arr[m_ctrlBlock->size], 0));
        setReadWriteDeviceVirtMemory(m_deviceId, m_ctrlBlock->ptr + offset, m_chunkSize);
        ++m_ctrlBlock->size;

        return true;
    }

    DMT_CPU size_t BuddyMemoryResource::minOrder() const
    {
        size_t const lzcntMinSize = std::countl_zero(m_chunkSize);
        size_t const lzcntMaxSize = std::countl_zero(m_maxPoolSize);
        return lzcntMinSize - lzcntMaxSize;
    }

    DMT_CPU size_t BuddyMemoryResource::blockToOrder(size_t size) const
    { // assumptions: size is already a POT between m_minBlockSize and m_chunkSize
        // should return: minOrder()     for size == m_chunkSize
        // should return: minOrder() + n for size == m_chunkSize >> n
        // Ensure the size is a power of two and within valid bounds
        assert(size >= m_minBlockSize && size <= m_chunkSize);
        assert((size & (size - 1)) == 0); // Check if size is a power of two

        // Calculate the minimum order
        size_t const minOrder = std::countl_zero(m_chunkSize) - std::countl_zero(static_cast<size_t>(m_minBlockSize));

        // Calculate the order for the given size
        size_t const order = minOrder + (std::countl_zero(m_chunkSize) - std::countl_zero(size));

        return order;
    }

    DMT_CPU size_t BuddyMemoryResource::alignUpToBlock(size_t size) const
    {
        size_t alignedSize = m_minBlockSize;
        while (alignedSize < size)
        {
            alignedSize <<= 1; // Double the size until it can accommodate 'size'.
        }
        return alignedSize;
    }

    DMT_CPU void BuddyMemoryResource::split(size_t order, size_t nodeIndex)
    { // shared_lock on m_ctx, lock_guard on transactionInFlight acquired by caller
        // Calculate raw array index of the node
        size_t const index = ((1ULL << order) - 1) + nodeIndex;

        // Ensure the node is valid
        assert(index < m_ctrlBlock->allocationBitmap.size());
        assert(m_ctrlBlock->allocationBitmap[index] == eFree || m_ctrlBlock->allocationBitmap[index] == eHasChildren);

        // Mark the current node as having children
        m_ctrlBlock->allocationBitmap[index] = eHasChildren;

        // Calculate the indices of the two children
        size_t const leftChildIndex  = 2 * index + 1;
        size_t const rightChildIndex = 2 * index + 2;

        // Ensure indices are within bounds
        assert(leftChildIndex < m_ctrlBlock->allocationBitmap.size());
        assert(rightChildIndex < m_ctrlBlock->allocationBitmap.size());

        // Mark both children as free
        m_ctrlBlock->allocationBitmap[leftChildIndex]  = eFree;
        m_ctrlBlock->allocationBitmap[rightChildIndex] = eFree;
    }

    DMT_CPU bool BuddyMemoryResource::coalesce(size_t parentIndex)
    { // shared_lock on m_ctx, lock_guard on transactionInFlight acquired by caller
        // Ensure the parent is valid and has children
        assert(parentIndex < m_ctrlBlock->allocationBitmap.size());
        assert(m_ctrlBlock->allocationBitmap[parentIndex] == eHasChildren);

        // Calculate indices of the children
        size_t const leftChildIndex  = 2 * parentIndex + 1;
        size_t const rightChildIndex = 2 * parentIndex + 2;

        // Ensure children indices are within bounds
        assert(leftChildIndex < m_ctrlBlock->allocationBitmap.size());
        assert(rightChildIndex < m_ctrlBlock->allocationBitmap.size());

        // Check if this parent level is valid for coalescing
        size_t const parentLevel = minOrder() - (std::countl_zero(parentIndex + 1) -
                                                 std::countl_zero(m_ctrlBlock->allocationBitmap.size()));
        if (parentLevel < minOrder())
        {
            return false; // Stop coalescing if parent level is below minOrder
        }

        // Check if both children are free
        if (m_ctrlBlock->allocationBitmap[leftChildIndex] == eFree && m_ctrlBlock->allocationBitmap[rightChildIndex] == eFree)
        {
            // Mark the parent as free
            m_ctrlBlock->allocationBitmap[parentIndex] = eFree;

            // Clear the children (optional, for debugging/clarity)
            m_ctrlBlock->allocationBitmap[leftChildIndex]  = eInvalid;
            m_ctrlBlock->allocationBitmap[rightChildIndex] = eInvalid;

            return true; // Coalescing succeeded
        }

        return false; // Coalescing not possible
    }

    BuddyMemoryResource::~BuddyMemoryResource()
    {
#if defined(__CUDA_ARCH__)
        // you shouldn't be here
#else
        cleanup();
#endif
    }

    void* BuddyMemoryResource::do_allocate(size_t _Bytes, size_t _Align)
    {
#if defined(__CUDA_ARCH__)
        return nullptr;
#else
        if (_Bytes > m_chunkSize)
            return nullptr;

        std::shared_lock slk{m_mtx};
        size_t           alignedSize = alignUpToBlock(_Bytes);
        size_t           level       = blockToOrder(alignedSize);

        std::lock_guard spinGuard{m_ctrlBlock->transactionInFlight};
        // Traverse the allocation bitmap to find a suitable free block
        while (true)
        {
            bool needsGrowth = false;
            for (size_t currentLevel = level; !needsGrowth && currentLevel <= minOrder(); ++currentLevel)
            {
                size_t const numNodesAtLevel = 1ULL << currentLevel;

                // Loop through nodes at the current level
                for (size_t nodeIdx = 0; !needsGrowth && nodeIdx < numNodesAtLevel; ++nodeIdx)
                {
                    size_t const index = ((1ULL << currentLevel) - 1) + nodeIdx;

                    // Check if the node is free
                    if (m_ctrlBlock->allocationBitmap[index] == eFree)
                    {
                        // Split the block as needed to reach the desired level
                        while (currentLevel > level)
                        {
                            --currentLevel;
                            split(currentLevel, nodeIdx / 2); // Split the parent
                            nodeIdx *= 2;                     // Move to the left child
                        }

                        // Mark the node as allocated
                        m_ctrlBlock->allocationBitmap[index] = eAllocated;

                        // Compute the pointer to the allocated memory
                        size_t const offset = nodeIdx * (m_chunkSize >> currentLevel);
                        CUdeviceptr  ptr    = m_ctrlBlock->ptr + offset;
                        CUdeviceptr  limit  = m_ctrlBlock->ptr + m_ctrlBlock->size * m_chunkSize;
                        if (ptr < limit)
                            return std::bit_cast<void*>(ptr);
                        else
                        {
                            needsGrowth = true;
                            break;
                        }
                    }
                }
            }

            // If no suitable block was found, grow the pool if possible
            if (!grow())
                return nullptr;
        }
#endif
    }

    void BuddyMemoryResource::do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
#if defined(__CUDA_ARCH__)
        // do nothing
#else
        std::shared_lock slk{m_mtx};
        // Align the size to the nearest power-of-two block
        _Bytes = alignUpToBlock(_Bytes);

        // Determine the level corresponding to the block size
        size_t const level = blockToOrder(_Bytes);

        // Compute the offset of the block relative to the start of the memory pool
        size_t const offset = std::bit_cast<uintptr_t>(_Ptr) - m_ctrlBlock->ptr;

        // Calculate the index of the block in the allocation bitmap
        size_t const nodeIdx = offset / (m_chunkSize >> level);
        size_t const index   = ((1ULL << level) - 1) + nodeIdx;

        // Acquire a lock for thread safety
        std::lock_guard lk{m_ctrlBlock->transactionInFlight};

        // Mark the block as free
        assert(m_ctrlBlock->allocationBitmap[index] == eAllocated);
        m_ctrlBlock->allocationBitmap[index] = eFree;

        // Attempt to coalesce blocks up the tree
        size_t parentIndex = (index - 1) / 2;
        while (parentIndex >= ((1ULL << (minOrder() - 1)) - 1) && coalesce(parentIndex))
        {
            parentIndex = (parentIndex - 1) / 2; // Move up to the parent
        }
#endif
    }

    inline bool BuddyMemoryResource::do_is_equal(memory_resource const& that) const noexcept
    { // TODO better
        BuddyMemoryResource const* other = dynamic_cast<BuddyMemoryResource const*>(&that);
        if (!other)
            return false;
        // if control block is the same, then they should be on the same device
        assert((m_ctrlBlock == other->m_ctrlBlock && m_device == other->m_device) || m_device != other->m_device);
        return m_device == other->m_device && m_ctrlBlock == other->m_ctrlBlock;
    }

    // Memory Resouce Boilerplate -------------------------------------------------------------------------------------
    DMT_CPU_GPU void switchOnMemoryResource(EMemoryResourceType eAlloc, BaseMemoryResource* p, size_t* sz, bool destroy, void* ctorParam)
    {
        EMemoryResourceType category = extractCategory(eAlloc);
        EMemoryResourceType type     = extractType(eAlloc);
        switch (category)
        {
            using enum EMemoryResourceType;
            case eHost:
                switch (type)
                {
                    case ePool:
                        if (p)
                            if (destroy)
                                std::destroy_at(std::bit_cast<HostPoolResource*>(p));
                            else
                                std::construct_at(std::bit_cast<HostPoolResource*>(p));
                        else if (sz)
                            *sz = sizeof(HostPoolResource);
                        break;
                    case eHostToDevMemMap:
                        if (p)
                            if (destroy)
                                std::destroy_at(std::bit_cast<BuddyMemoryResource*>(p));
                            else
                                std::construct_at(std::bit_cast<BuddyMemoryResource*>(p),
                                                  *std::bit_cast<BuddyResourceSpec*>(ctorParam));
                        else if (sz)
                            *sz = sizeof(BuddyMemoryResource);
                        break;
                }
                break;
            case eDevice:
                switch (type)
                {
                    case eCudaMalloc:
                        if (p)
                            if (destroy)
                                std::destroy_at(std::bit_cast<CudaMallocResource*>(p));
                            else
                                std::construct_at(std::bit_cast<CudaMallocResource*>(p));
                        else if (sz)
                            *sz = sizeof(CudaMallocResource);
                }
                break;
            case eAsync:
                switch (type)
                {
                    case eCudaMallocAsync:
                        if (p)
                            if (destroy)
                                std::destroy_at(std::bit_cast<CudaMallocAsyncResource*>(p));
                            else
                                std::construct_at(std::bit_cast<CudaMallocAsyncResource*>(p));
                        else if (sz)
                            *sz = sizeof(CudaMallocAsyncResource);
                }
                break;
            case eUnified:
                switch (type)
                {
                    case eCudaMallocManaged:
                        if (p)
                            if (destroy)
                                std::destroy_at(std::bit_cast<UnifiedMemoryResource*>(p));
                            else
                                std::construct_at(std::bit_cast<UnifiedMemoryResource*>(p));
                        else if (sz)
                            *sz = sizeof(UnifiedMemoryResource);
                        break;
                }
                break;
        }
    }

    DMT_CPU_GPU size_t sizeForMemoryResouce(EMemoryResourceType eAlloc)
    {
        size_t ret = 0;
        switchOnMemoryResource(eAlloc, nullptr, &ret, true, nullptr);
        return ret;
    }

    DMT_CPU_GPU BaseMemoryResource* constructMemoryResourceAt(void* ptr, EMemoryResourceType eAlloc, void* ctorParam)
    {
        BaseMemoryResource* p = std::bit_cast<BaseMemoryResource*>(ptr);
        switchOnMemoryResource(eAlloc, p, nullptr, false, ctorParam);
        return p;
    }

    DMT_CPU_GPU void destroyMemoryResouceAt(BaseMemoryResource* p, EMemoryResourceType eAlloc)
    {
        switchOnMemoryResource(eAlloc, p, nullptr, true, nullptr);
    }

    DMT_CPU_GPU EMemoryResourceType categoryOf(BaseMemoryResource* allocator)
    {
        if (dynamic_cast<CudaAsyncMemoryReosurce*>(allocator))
            return EMemoryResourceType::eAsync;
        else if (dynamic_cast<DeviceMemoryReosurce*>(allocator))
            return EMemoryResourceType::eDevice;
        else if (dynamic_cast<UnifiedMemoryResource*>(allocator))
            return EMemoryResourceType::eUnified;
        else if (dynamic_cast<std::pmr::memory_resource*>(allocator))
            return EMemoryResourceType::eHost;
        assert(false);
        return EMemoryResourceType::eHost;
    }

    DMT_CPU_GPU bool isDeviceAllocator(BaseMemoryResource* allocator)
    {
        if (dynamic_cast<CudaAsyncMemoryReosurce*>(allocator))
            return true;
        else if (dynamic_cast<DeviceMemoryReosurce*>(allocator))
            return true;
        else if (dynamic_cast<UnifiedMemoryResource*>(allocator))
            return true;
        else if (dynamic_cast<std::pmr::memory_resource*>(allocator))
            return false;

        return false;
    }

    DMT_CPU_GPU bool isHostAllocator(BaseMemoryResource* allocator)
    {
        if (dynamic_cast<CudaAsyncMemoryReosurce*>(allocator))
            return false;
        else if (dynamic_cast<DeviceMemoryReosurce*>(allocator))
            return false;
        else if (dynamic_cast<UnifiedMemoryResource*>(allocator))
            return true;
        else if (dynamic_cast<std::pmr::memory_resource*>(allocator))
            return true;

        return false;
    }

    DMT_CPU_GPU void* allocateFromCategory(BaseMemoryResource* allocator, size_t sz, size_t align, CudaStreamHandle stream)
    {
        if (auto* a = dynamic_cast<CudaAsyncMemoryReosurce*>(allocator); a)
        {
#if defined(__CUDA_ARCH__)
            return a->allocate(sz, align);
#else
            if (stream != noStream)
            {
                cuda::stream_ref streamref = streamRefFromHandle(stream);
                return a->allocate_async(sz, align, streamref);
            }
            else
                return a->allocate(sz, align);
#endif
        }
        else if (auto* a = dynamic_cast<DeviceMemoryReosurce*>(allocator); a)
            return a->allocate(sz, align);
        else if (auto* a = dynamic_cast<UnifiedMemoryResource*>(allocator); a)
            return a->allocate(sz, align);
        else if (auto* a = dynamic_cast<std::pmr::memory_resource*>(allocator); a)
            return a->allocate(sz, align);

        assert(false);
        return nullptr;
    }

    DMT_CPU_GPU void freeFromCategory(BaseMemoryResource* allocator, void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        if (auto* a = dynamic_cast<CudaAsyncMemoryReosurce*>(allocator); a)
        {
#if defined(__CUDA_ARCH__)
            a->deallocate(ptr, sz, align);
#else
            if (stream != noStream)
            {
                cuda::stream_ref streamref = streamRefFromHandle(stream);
                a->deallocate_async(ptr, sz, align, streamref);
            }
            else
                a->deallocate(ptr, sz, align);
#endif
        }
        else if (auto* a = dynamic_cast<DeviceMemoryReosurce*>(allocator); a)
            a->deallocate(ptr, sz, align);
        else if (auto* a = dynamic_cast<UnifiedMemoryResource*>(allocator); a)
            a->deallocate(ptr, sz, align);
        else if (auto* a = dynamic_cast<std::pmr::memory_resource*>(allocator); a)
            a->deallocate(ptr, sz, align);
    }

    // BaseDeviceContainer --------------------------------------------------------------------------------------------
    // TODO: use cuda::atomic_ref instead of atomic primitives. Reference: https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html
    DMT_CPU_GPU void BaseDeviceContainer::lockForRead() const
    {
#if defined(__CUDA_ARCH__)
        // Wait until no writer is active
        while (atomicAdd(&m_writeCount, 0) > 0)
        {
            // Spin-wait
        }
        // Increment reader count
        atomicAdd(&m_readCount, 1);
#else
        std::atomic_ref<int> writeRef(m_writeCount);
        std::atomic_ref<int> readRef(m_readCount);
        // Wait until no writer is active
        while (writeRef.load(std::memory_order_acquire) > 0)
        {
            // Spin-wait
        }
        // Increment reader count
        readRef.fetch_add(1, std::memory_order_acquire);
#endif
    }

    DMT_CPU_GPU void BaseDeviceContainer::unlockForRead() const
    {
#if defined(__CUDA_ARCH__)
        // Decrement reader count
        atomicSub(&m_readCount, 1);
#else
        std::atomic_ref<int> readRef(m_readCount);
        // Decrement reader count
        readRef.fetch_sub(1, std::memory_order_release);
#endif
    }

    DMT_CPU_GPU void BaseDeviceContainer::lockForWrite() const
    {
#if defined(__CUDA_ARCH__)
        // Wait until no reader or writer is active
        while (atomicAdd(&m_readCount, 0) > 0 || atomicAdd(&m_writeCount, 0) > 0)
        {
            // Spin-wait
        }
        // Increment writer count
        atomicAdd(&m_writeCount, 1);
#else
        std::atomic_ref<int> writeRef(m_writeCount);
        std::atomic_ref<int> readRef(m_readCount);
        // Wait until no reader or writer is active
        while (readRef.load(std::memory_order_acquire) > 0 || writeRef.load(std::memory_order_acquire) > 0)
        {
            // Spin-wait
        }
        // Increment writer count
        writeRef.fetch_add(1, std::memory_order_acquire);
#endif
    }

    DMT_CPU_GPU void BaseDeviceContainer::unlockForWrite() const
    {
#if defined(__CUDA_ARCH__)
        // Decrement writer count
        atomicSub(&m_writeCount, 1);
#else
        std::atomic_ref<int> writeRef(m_writeCount);
        // Decrement writer count
        writeRef.fetch_sub(1, std::memory_order_release);
#endif
    }

    DMT_CPU_GPU void BaseDeviceContainer::waitWriter() const
    {
#if defined(__CUDA_ARCH__)
        // Spin-wait until no writer is active
        while (atomicAdd(&m_writeCount, 0) > 0)
        {
            // Spin-wait
        }
#else
        std::atomic_ref<int> writeRef(m_writeCount);
        // Spin-wait until no writer is active
        while (writeRef.load(std::memory_order_acquire) > 0)
        {
            // Spin-wait
        }
#endif
    }

    // DynaArray ------------------------------------------------------------------------------------------------------
    DMT_CPU_GPU DynaArray::DynaArray(DynaArray const& other) :
    BaseDeviceContainer(other.m_resource, other.stream),
    m_elemSize(other.m_elemSize)
    {
        other.lockForRead();
        m_resource = other.m_resource;
        stream     = other.stream;
        m_elemSize = other.m_elemSize;
        reserve(other.m_size);
        lockForWrite();
        copyFrom(other);
        unlockForWrite();
        other.unlockForRead();
    }

    DMT_CPU_GPU DynaArray::DynaArray(DynaArray&& other) noexcept : BaseDeviceContainer(other.m_resource, other.stream)
    {
        other.lockForWrite();
        m_resource = std::exchange(other.m_resource, nullptr);
        stream     = other.stream;
        m_head     = std::exchange(other.m_head, nullptr);
        m_size     = std::exchange(other.m_size, 0);
        m_capacity = std::exchange(other.m_capacity, 0);
        m_elemSize = other.m_elemSize;
        other.unlockForWrite();
    }

    DMT_CPU_GPU void DynaArray::reserve(size_t newCapacity, bool lock)
    {
        if (lock)
            lockForWrite();
        if (newCapacity <= m_capacity)
            return;

        void* newHead = allocateFromCategory(m_resource, newCapacity * m_elemSize, alignof(std::max_align_t), stream);
        if (m_head)
        {
            std::memcpy(newHead, m_head, m_size * m_elemSize);
            freeFromCategory(m_resource, m_head, m_size * m_elemSize, alignof(std::max_align_t), stream);
        }
        m_head     = newHead;
        m_capacity = newCapacity;
        if (lock)
            unlockForWrite();
    }

    DMT_CPU_GPU void DynaArray::clear(bool lock) noexcept
    {
        if (lock)
            lockForWrite();
        if (m_head)
        {
            freeFromCategory(m_resource, m_head, m_size * m_elemSize, alignof(std::max_align_t), stream);
            m_head = nullptr;
        }
        if (lock)
            unlockForWrite();
    }

    DMT_CPU_GPU bool DynaArray::push_back(void const* pValue, bool srcHost, bool lock)
    {
        bool ret = true;
        if (lock)
            lockForWrite();

        if (m_size >= m_capacity)
            reserve(m_capacity > 0 ? m_capacity >> 1 : 1, false);

        void* dest = std::bit_cast<void*>(std::bit_cast<uintptr_t>(m_head) + m_size * m_elemSize);

#if defined(__CUDA_ARCH__)
        if (srcHost) // error!
            ret = false;
        else
            std::memcpy(dest, pValue, m_elemSize);
#else
        cudaMemcpyKind kind = isDeviceAllocator(m_resource)
                                  ? (srcHost ? ::cudaMemcpyHostToDevice : ::cudaMemcpyDeviceToDevice)
                                  : (srcHost ? ::cudaMemcpyHostToHost : ::cudaMemcpyDeviceToHost);
        cudaError_t    res  = cudaMemcpy(dest, pValue, m_elemSize, kind);
        if (res != ::cudaSuccess)
            ret = false;
#endif
        if (ret)
            ++m_size;

        if (lock)
            unlockForWrite();
        return ret;
    }

    DMT_CPU_GPU void DynaArray::pop_back(bool lock)
    {
        if (lock)
            lockForWrite();

        if (m_size != 0)
            --m_size;

        if (lock)
            unlockForWrite();
    }

    // assumes you already locked for read

    DMT_CPU_GPU void const* DynaArray::at(size_t index) const
    {
        assert(index < m_size);
#if defined(__CUDA_ARCH__)
        if (isDeviceAllocator(m_resource))
            return std::bit_cast<void*>(std::bit_cast<uintptr_t>(m_head) + index * m_elemSize);
        else
            return nullptr;
#else
        if (isHostAllocator(m_resource))
            return std::bit_cast<void*>(std::bit_cast<uintptr_t>(m_head) + index * m_elemSize);
        else
        {
            assert(false);
            return nullptr;
        }
#endif
    }

    DMT_CPU_GPU void DynaArray::copyFrom(DynaArray const& other)
    {
        if (other.m_size > 0)
        {
            std::memcpy(m_head, other.m_head, other.m_size * other.m_elemSize);
            m_size = other.m_size;
        }
    }

    DMT_CPU_GPU DynaArray& DynaArray::operator=(DynaArray const& other)
    {
        if (this != &other)
        {
            lockForWrite();
            other.lockForWrite();
            assert(m_elemSize == other.m_elemSize);

            clear();
            reserve(other.m_size, false);
            copyFrom(other);

            other.unlockForWrite();
            unlockForWrite();
        }
        return *this;
    }

    DMT_CPU_GPU DynaArray& DynaArray::operator=(DynaArray&& other) noexcept
    {
        if (this != &other)
        {
            lockForWrite();
            other.lockForWrite();
            assert(m_elemSize == other.m_elemSize);

            m_size     = std::exchange(other.m_size, 0);
            m_capacity = std::exchange(other.m_capacity, 0);
            m_head     = std::exchange(other.m_head, nullptr);

            other.unlockForWrite();
            unlockForWrite();
        }
        return *this;
    }

    DMT_CPU_GPU DynaArray::~DynaArray() noexcept { clear(); }

    DMT_CPU bool DynaArray::copyToHostSync(void* /*DMT_RESTRICT*/ dest, bool lock) const
    {
        if (!isDeviceAllocator(m_resource))
        {
            assert(false);
            return false;
        }

        bool ret = true;
        if (lock)
            lockForRead();

        ret = ::cudaSuccess == cudaMemcpy(dest, m_head, m_size * m_elemSize, ::cudaMemcpyDeviceToHost);

        if (lock)
            unlockForRead();

        return ret;
    }

    DMT_CPU_GPU size_t DynaArray::size(bool lock) const
    {
        size_t ret = 0;
        if (lock)
            lockForRead();
        ret = m_size;
        if (lock)
            unlockForRead();
        return ret;
    }
} // namespace dmt
