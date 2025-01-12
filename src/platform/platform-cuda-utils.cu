#define DMT_INTERFACE_AS_HEADER
#undef DMT_NEEDS_MODULE
// CANNOT USE os-utils
#include "platform-cuda-utils.cuh"
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

    __host__ CudaStreamHandle newStream()
    {
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) == ::cudaSuccess)
            return std::bit_cast<CudaStreamHandle>(stream);
        else
            return noStream;
    }

    __host__ CUdevice currentDeviceHandle()
    {
        CUdevice    ret;
        int32_t     deviceId = 0;
        cudaError_t err      = cudaGetDevice(&deviceId);
        assert(err == ::cudaSuccess);
        CUresult err1 = cuDeviceGet(&ret, deviceId);
        assert(err1 == ::CUDA_SUCCESS);
        return ret;
    }

    __host__ void deleteStream(CudaStreamHandle stream)
    {
        if (stream != noStream && stream != 0)
        {
            cudaStream_t cStream = std::bit_cast<cudaStream_t>(stream);
            cudaError_t  err     = cudaStreamDestroy(cStream);
            assert(err == ::cudaSuccess);
        }
    }

    __host__ CUDAHelloInfo cudaHello(MemoryContext* mctx)
    {
        CUDAHelloInfo ret;
        ret.cudaCapable = false;
        // Force cuda context lazy initialization (driver and runtime interop:
        // https://stackoverflow.com/questions/60132426/how-can-i-mix-cuda-driver-api-with-cuda-runtime-api
        int32_t count = 0;
        if (cudaGetDeviceCount(&count) != ::cudaSuccess || count <= 0)
        {
            if (mctx)
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
            if (mctx)
                mctx->pctx.error("Couldn't find any CUDA device which suits the desired requiresments");
            return ret;
        }
        ret.cudaCapable = true;
        ret.device      = device;

        cudaDeviceProp actualProps{};
        if (cudaGetDeviceProperties(&actualProps, device) != ::cudaSuccess)
        {
            if (mctx)
                mctx->pctx.error("Couldn't get device {} properties", {device});
            return ret;
        }
        if (mctx)
        {
            mctx->pctx.log("Chosed Device: {} ({})", {std::string_view{actualProps.name}, device});
            mctx->pctx.log("Compute Capability: {}.{}", {actualProps.major, actualProps.minor});
        }
        assert(actualProps.managedMemory && actualProps.canMapHostMemory);

        ret.warpSize = actualProps.warpSize;

        // forrce CUDA context initialization (after this, you can use the CUDA driver API)
        // the context, if needed, can be fetched with `cuCtxGetCurrent`
        if (cudaFree(nullptr) != ::cudaSuccess)
        {
            if (mctx)
                mctx->pctx.error("Couldn't initialize CUDA context");
            ret.cudaCapable = false;
            return ret;
        }
        size_t totalBytes = ret.totalMemInBytes = 0;
        if (cudaMemGetInfo(nullptr, &totalBytes) != ::cudaSuccess)
            if (mctx)
                mctx->pctx.error("Couldn't get the total Memory in bytes of the device");
            else
            {
                ret.totalMemInBytes = totalBytes;
                if (mctx)
                    mctx->pctx.log("Total Device Memory: {}", {ret.totalMemInBytes});
            }

        if (actualProps.canMapHostMemory)
        { // all flags starts with `cudaDevice*`
            if (cudaSetDeviceFlags(cudaDeviceMapHost) != ::cudaSuccess)
            {
                if (mctx)
                    mctx->pctx.error("Failed to enable device flags for pin map host memory");
            }
        }

        // check current device support for for CU_DEVICE_ATTRIBUTE_MEMORY_POOLS and CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY
        CUdevice deviceHandle;
        CUresult res;
        res = cuDeviceGet(&deviceHandle, device);
        assert(res == ::CUDA_SUCCESS);
        int32_t support = 0;
        res             = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        assert(support <= 1);
        ret.supportsMemoryPools = support;

        // support for `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` and related
        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        assert(support <= 1);
        ret.supportsVirtualMemory = support;

        // various inofration
        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.perMultiprocessorMaxBlocks = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.perMultiprocessor.maxRegisters = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.perMultiprocessor.maxSharedMemory = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.L2CacheBytes = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.multiprocessorCount = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.perBlock.maxRegisters = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.perBlock.maxSharedMemory = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.constantMemoryBytes = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.perBlockMaxThreads = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.maxBlockDim.x = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.maxBlockDim.y = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.maxBlockDim.z = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.maxGridDim.x = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
        ret.maxGridDim.y = support;

        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, deviceHandle);
        assert(res == ::CUDA_SUCCESS);
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

    __device__ int32_t globalThreadIndex()
    {
        int32_t const column          = threadIdx.x;
        int32_t const row             = threadIdx.y;
        int32_t const aisle           = threadIdx.z;
        int32_t const threads_per_row = blockDim.x;                  //# threads in x direction aka row
        int32_t const threads_per_aisle = (blockDim.x * blockDim.y); //# threads in x and y direction for total threads per aisle

        int32_t const threads_per_block = (blockDim.x * blockDim.y * blockDim.z);
        int32_t const rowOffset         = (row * threads_per_row);     //how many rows to push out offset by
        int32_t const aisleOffset       = (aisle * threads_per_aisle); // how many aisles to push out offset by

        //S32_t constecond section locates and caculates block offset withing the grid
        int32_t const blockColumn    = blockIdx.x;
        int32_t const blockRow       = blockIdx.y;
        int32_t const blockAisle     = blockIdx.z;
        int32_t const blocks_per_row = gridDim.x;                 //# blocks in x direction aka blocks per row
        int32_t const blocks_per_aisle = (gridDim.x * gridDim.y); // # blocks in x and y direction for total blocks per aisle
        int32_t const blockRowOffset   = (blockRow * blocks_per_row);     // how many rows to push out block offset by
        int32_t const blockAisleOffset = (blockAisle * blocks_per_aisle); // how many aisles to push out block offset by
        int32_t const blockId          = blockColumn + blockRowOffset + blockAisleOffset;

        int32_t const blockOffset = (blockId * threads_per_block);

        int32_t const gid = (blockOffset + aisleOffset + rowOffset + column);
        return gid;
    }

    __device__ int32_t warpWideThreadIndex()
    {
        int32_t id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        return id % warpSize;
    }

    __host__ void* cudaAllocate(size_t sz)
    {
        void*       tmp = nullptr;
        cudaError_t err = cudaMallocManaged(&tmp, sz);
        if (err != ::cudaSuccess)
            return nullptr;
        return tmp;
    }

    __host__ __device__ void cudaDeallocate(void* ptr, size_t sz)
    {
        if (ptr)
            cudaFree(ptr);
    }

    __host__ UnifiedMemoryResource::UnifiedMemoryResource() :
        BaseMemoryResource(makeMemResId(EMemoryResourceType::eUnified, EMemoryResourceType::eCudaMallocManaged))
    {
        m_host.allocateBytes = UnifiedMemoryResource::allocateBytes;
        m_host.freeBytes = UnifiedMemoryResource::freeBytes;
        m_host.allocateBytesAsync = UnifiedMemoryResource::allocateBytesAsync;
        m_host.freeBytesAsync = UnifiedMemoryResource::freeBytesAsync;
        m_host.deviceHasAccess = UnifiedMemoryResource::deviceHasAccess;
        m_host.hostHasAccess = UnifiedMemoryResource::hostHasAccess;
        initTable<<<1, 32 >>>(*this);
        cudaDeviceSynchronize();
    }

    __host__ void* UnifiedMemoryResource::do_allocate(size_t sz, size_t align) {
        return UnifiedMemoryResource::allocateBytes(this, sz, align);
    }

    __host__ void UnifiedMemoryResource::do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
        UnifiedMemoryResource::freeBytes(this, _Ptr, _Bytes, _Align);
    }

    __host__ bool UnifiedMemoryResource::do_is_equal(memory_resource const& _That) const noexcept { return true; }

    __host__ __device__ void* UnifiedMemoryResource::allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align)
    {
#if defined(__CUDA_ARCH__)
        // you shouldn't be here
        return nullptr;
#else
        return cudaAllocate(sz);
#endif
    }

    __host__ __device__ void UnifiedMemoryResource::freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align)
    {
#if !defined(__CUDA_ARCH__)
        cudaDeallocate(ptr, sz);
#endif
    }

    __host__ void* UnifiedMemoryResource::allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
        return nullptr;
    }

    __host__ void UnifiedMemoryResource::freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
    }

    __host__ __device__ bool UnifiedMemoryResource::deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID)
    {
        int32_t     currDev;
        cudaError_t cudaErr = cudaGetDevice(&currDev);
        assert(cudaErr == ::cudaSuccess);
        int32_t supported = 0;
        cudaErr           = cudaDeviceGetAttribute(&supported, ::cudaDevAttrManagedMemory, currDev);
        assert(cudaErr == ::cudaSuccess);
        return supported != 0;
    }

    __host__ __device__ bool UnifiedMemoryResource::hostHasAccess(BaseMemoryResource const* pAlloc) { return true; }

    // Memory Resouce Interfaces --------------------------------------------------------------------------------------
    __host__ __device__ void* BaseMemoryResource::tryAllocateAsync(size_t sz, size_t align, CudaStreamHandle stream)
    {
#if defined(__CUDA_ARCH__)
        return allocateBytes(sz, align);
#else
        if (stream != noStream && categoryOf(this) == EMemoryResourceType::eAsync)
            return allocateBytesAsync(sz, align, stream);
        else
            return allocateBytes(sz, align);
#endif
    }

    __host__ __device__ void BaseMemoryResource::tryFreeAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
#if defined(__CUDA_ARCH__)
        return freeBytes(ptr, sz, align);
#else
        if (stream != noStream && categoryOf(this) == EMemoryResourceType::eAsync)
            return freeBytesAsync(ptr, sz, align, stream);
        else
            return freeBytes(ptr, sz, align);
#endif
    }

    __host__ __device__ void* BaseMemoryResource::allocateBytes(size_t sz, size_t align) {
#if defined(__CUDA_ARCH__)
        assert(m_device.allocateBytes);
        return m_device.allocateBytes(this, sz, align);
#else
        assert(m_host.allocateBytes);
        return m_host.allocateBytes(this, sz, align);
#endif
    }

    __host__ __device__ void BaseMemoryResource::freeBytes(void* ptr, size_t sz, size_t align) {
#if defined(__CUDA_ARCH__)
        assert(m_device.freeBytes);
        m_device.freeBytes(this, ptr, sz, align);
#else
        assert(m_host.freeBytes);
        m_host.freeBytes(this, ptr, sz, align);
#endif
    }

    __host__ void* BaseMemoryResource::allocateBytesAsync(size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(m_host.allocateBytesAsync);
        return m_host.allocateBytesAsync(this, sz, align, stream);
    }

    __host__ void BaseMemoryResource::freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) {
        assert(m_host.freeBytesAsync);
        m_host.freeBytesAsync(this, ptr, sz, align, stream);
    }

    __host__ __device__ bool BaseMemoryResource::deviceHasAccess(int32_t deviceID) const {
#if defined(__CUDA_ARCH__)
        assert(m_device.deviceHasAccess);
        return m_device.deviceHasAccess(this, deviceID);
#else
        assert(m_host.deviceHasAccess);
        return m_host.deviceHasAccess(this, deviceID);
#endif
    }

    __host__ __device__ bool BaseMemoryResource::hostHasAccess() const {
#if defined(__CUDA_ARCH__)
        assert(m_device.hostHasAccess);
        return m_device.hostHasAccess(this);
#else
        assert(m_host.hostHasAccess);
        return m_host.hostHasAccess(this);
#endif
    }

    // Memory Allocation Helpers --------------------------------------------------------------------------------------
    size_t getDeviceAllocationGranularity(int32_t deviceId, CUmemAllocationProp* outProp)
    {
        CUmemAllocationProp prop = {};
        prop.type                = ::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = ::CU_MEM_LOCATION_TYPE_DEVICE; // allocate memory on device whose id is given
        prop.location.id         = deviceId;
        size_t   granularity     = 0;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, ::CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        assert(res == ::CUDA_SUCCESS);
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

        CUresult res = cuMemSetAccess(ptr, size, &accessDesc, 1);
        assert(res == ::CUDA_SUCCESS);
    }

    // Memory Resouce Implementations ---------------------------------------------------------------------------------
    __host__ __device__ void* CudaMallocResource::allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align) {
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, sz) != ::cudaSuccess)
            return nullptr;
        return tmp;
    }
    __host__ __device__ void  CudaMallocResource::freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align) {
        cudaError_t err = cudaFree(ptr);
        assert(err == ::cudaSuccess);
    }
    __host__ void* CudaMallocResource::allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream) {
        assert(false);
        return nullptr;
    }
    __host__ void CudaMallocResource::freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream) {
        assert(false);
    }

    __host__ CudaMallocResource::CudaMallocResource() : DeviceMemoryReosurce(EMemoryResourceType::eCudaMalloc) {
        m_host.allocateBytes = CudaMallocResource::allocateBytes;
        m_host.freeBytes = CudaMallocResource::freeBytes;
        m_host.allocateBytesAsync = CudaMallocResource::allocateBytesAsync;
        m_host.freeBytesAsync = CudaMallocResource::freeBytesAsync;
        m_host.deviceHasAccess = CudaMallocResource::deviceHasAccess;
        m_host.hostHasAccess = CudaMallocResource::hostHasAccess;
        initTable<<<1, 32>>>(*this);
        cudaDeviceSynchronize();
    }
    __host__ __device__ void* CudaMallocResource::allocate(size_t sz, size_t align)
    {
        return CudaMallocResource::allocateBytes(this, sz, align);
    }
    __host__ __device__ void CudaMallocResource::deallocate(void* ptr, size_t sz, size_t align)
    {
        CudaMallocResource::freeBytes(this, ptr, sz, align);
    }


    __host__ __device__ void* CudaMallocAsyncResource::allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align) {
#if !defined(__CUDA_ARCH__)
        return reinterpret_cast<CudaMallocAsyncResource*>(pAlloc)->allocate(sz, align);
#else
        assert(false);
        return nullptr;
#endif
    }
    __host__ __device__ void  CudaMallocAsyncResource::freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align) {
#if !defined(__CUDA_ARCH__)
        reinterpret_cast<CudaMallocAsyncResource*>(pAlloc)->deallocate(ptr, sz, align);
#else
        assert(false);
#endif
    }
    __host__ void* CudaMallocAsyncResource::allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream) {
        return reinterpret_cast<CudaMallocAsyncResource*>(pAlloc)->allocate_async(sz, align, streamRefFromHandle(stream));
    }
    __host__ void     CudaMallocAsyncResource::freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream) {
        reinterpret_cast<CudaMallocAsyncResource*>(pAlloc)->deallocate_async(ptr, sz, align, streamRefFromHandle(stream));
    }

    __host__ CudaMallocAsyncResource::CudaMallocAsyncResource() : CudaAsyncMemoryReosurce(EMemoryResourceType::eCudaMallocAsync) {
        m_host.allocateBytes = CudaMallocAsyncResource::allocateBytes;
        m_host.freeBytes = CudaMallocAsyncResource::freeBytes;
        m_host.allocateBytesAsync = CudaMallocAsyncResource::allocateBytesAsync;
        m_host.freeBytesAsync = CudaMallocAsyncResource::freeBytesAsync;
        m_host.hostHasAccess = CudaMallocAsyncResource::hostHasAccess;
        m_host.deviceHasAccess = CudaMallocAsyncResource::deviceHasAccess;
        initTable<<<1, 32>>>(*this);
        cudaDeviceSynchronize();
    }

    void* CudaMallocAsyncResource::allocate(size_t _Bytes, [[maybe_unused]] size_t _Align)
    {
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, _Bytes) != ::cudaSuccess)
            return nullptr;
        return tmp;
    }
    void CudaMallocAsyncResource::deallocate(void* _Ptr, size_t _Bytes, [[maybe_unused]] size_t _Align)
    {
        cudaError_t err = cudaFree(_Ptr);
        assert(err == ::cudaSuccess);
    }
    __host__ void* CudaMallocAsyncResource::allocate_async(size_t sz, [[maybe_unused]] size_t align, cuda::stream_ref stream)
    {
        void* tmp = nullptr;
        if (cudaMallocAsync(&tmp, sz, stream.get()) != ::cudaSuccess)
            return nullptr;
        return tmp;
    }
    __host__ void CudaMallocAsyncResource::deallocate_async(
        void*                   ptr,
        [[maybe_unused]] size_t sz,
        [[maybe_unused]] size_t align,
        cuda::stream_ref        stream)
    {
        cudaError_t err = cudaFreeAsync(ptr, stream.get());
        assert(err == ::cudaSuccess);
    }

    __host__ BuddyMemoryResource::BuddyMemoryResource(BuddyResourceSpec const& input) :
    BaseMemoryResource(makeMemResId(EMemoryResourceType::eHost, EMemoryResourceType::eHostToDevMemMap)),
    m_deviceId(input.deviceId),
    m_minBlockSize(static_cast<uint32_t>(nextPOT(input.minBlockSize)))
    { 
        m_host.allocateBytes = BuddyMemoryResource::allocateBytes;
        m_host.freeBytes = BuddyMemoryResource::freeBytes;
        m_host.allocateBytesAsync = BuddyMemoryResource::allocateBytesAsync;
        m_host.freeBytesAsync = BuddyMemoryResource::freeBytesAsync;
        m_host.deviceHasAccess = BuddyMemoryResource::deviceHasAccess;
        m_host.hostHasAccess = BuddyMemoryResource::hostHasAccess;
        initTable<<<1, 32>>>(*this);
        cudaDeviceSynchronize();
        
        // TODO better? allocation functions without class? global context map {pid, idx -> ctx}
        assert(input.pHostMemRes);
        CUresult res = cuDeviceGet(&m_deviceHnd, m_deviceId);
        assert(res == ::CUDA_SUCCESS);
        int32_t support = 0;
        res             = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, m_deviceHnd);
        assert(res == ::CUDA_SUCCESS);
        if (support == 0)
        {
            if (input.pmctx)
                input.pmctx->pctx.error("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED not supported on the given device");
            std::abort();
        }
        res = cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, m_deviceHnd);
        assert(res == ::CUDA_SUCCESS);
        if (support == 0)
        {
            if (input.pmctx)
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
            if (input.pmctx)
                input.pmctx->pctx.error("Couldn't reserve host virtual address space for {} Bytes of metadata",
                                        {m_ctrlBlockReservedVMemBytes});
            std::abort();
        }
        // commit first page, construct control block, compute initial capacity
        if (!commitPhysicalMemory(reservedHostSpace, toUnderlying(EPageSize::e4KB)))
        {
            if (input.pmctx)
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
        res = cuMemAddressReserve(&m_ctrlBlock->ptr, m_maxPoolSize, 0 /*natural align*/, 0 /*hint address*/, 0 /*flags*/);
        assert(res == ::CUDA_SUCCESS);
        CUmemGenericAllocationHandle* arr = vlaStart();
        if (!allocateDevicePhysicalMemory(m_deviceId, m_chunkSize, arr[0]))
        {
            if (input.pmctx)
                input.pmctx->pctx.error("Couldn't allocate first Device memory chunk, device memory already exhausted");
            std::abort();
        }
        ++m_ctrlBlock->size;

        // map the first device memory chunk
        res = cuMemMap(m_ctrlBlock->ptr, m_chunkSize, 0, arr[0], 0);
        assert(res == ::CUDA_SUCCESS);
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
    }

    __host__ BuddyMemoryResource::BuddyMemoryResource(BuddyMemoryResource const& other) :
    BaseMemoryResource(makeMemResId(EMemoryResourceType::eHost, EMemoryResourceType::eHostToDevMemMap))
    {
        // shouldn't be necessary to lock `transactionInFlight`, cause no device allocation is needed here
        std::shared_lock slk{other.m_mtx};
        m_ctrlBlock                  = other.m_ctrlBlock;
        m_maxPoolSize                = other.m_maxPoolSize;
        m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
        m_chunkSize                  = other.m_chunkSize;
        m_deviceHnd                     = other.m_deviceHnd;
        m_deviceId                   = other.m_deviceId;
        m_minBlockSize               = other.m_minBlockSize;

        std::lock_guard lk{m_ctrlBlock->transactionInFlight};
        m_ctrlBlock->refCount.fetch_add(1, std::memory_order_seq_cst);
    }

    __host__ BuddyMemoryResource::BuddyMemoryResource(BuddyMemoryResource&& other) noexcept :
    BaseMemoryResource(makeMemResId(EMemoryResourceType::eHost, EMemoryResourceType::eHostToDevMemMap))
    {
        // shouldn't be necessary to lock `transactionInFlight`, cause no device allocation is needed here
        std::lock_guard lk{other.m_mtx};
        m_ctrlBlock                  = std::exchange(other.m_ctrlBlock, nullptr);
        m_maxPoolSize                = other.m_maxPoolSize;
        m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
        m_chunkSize                  = other.m_chunkSize;
        m_deviceHnd                     = other.m_deviceHnd;
        m_deviceId                   = other.m_deviceId;
        m_minBlockSize               = other.m_minBlockSize;
    }

    __host__ BuddyMemoryResource& BuddyMemoryResource::operator=(BuddyMemoryResource const& other)
    {
        if (*this != other)
        {
            cleanup();
            std::unique_lock lk{m_mtx, std::defer_lock};
            std::shared_lock slk{other.m_mtx, std::defer_lock};
            std::lock(lk, slk);

            m_ctrlBlock                  = other.m_ctrlBlock;
            m_maxPoolSize                = other.m_maxPoolSize;
            m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
            m_chunkSize                  = other.m_chunkSize;
            m_deviceHnd                     = other.m_deviceHnd;
            m_deviceId                   = other.m_deviceId;
            m_minBlockSize               = other.m_minBlockSize;

            std::lock_guard mlk{m_ctrlBlock->transactionInFlight};
            m_ctrlBlock->refCount.fetch_add(1, std::memory_order_seq_cst);
        }
        return *this;
    }

    __host__ BuddyMemoryResource& BuddyMemoryResource::operator=(BuddyMemoryResource&& other) noexcept
    {
        if (*this != other)
        {
            cleanup();
            std::unique_lock lk{m_mtx, std::defer_lock};
            std::unique_lock olk{other.m_mtx, std::defer_lock};
            std::lock(lk, olk);

            m_ctrlBlock                  = std::exchange(other.m_ctrlBlock, nullptr);
            m_maxPoolSize                = other.m_maxPoolSize;
            m_ctrlBlockReservedVMemBytes = other.m_ctrlBlockReservedVMemBytes;
            m_chunkSize                  = other.m_chunkSize;
            m_deviceHnd                     = other.m_deviceHnd;
            m_deviceId                   = other.m_deviceId;
            m_minBlockSize               = other.m_minBlockSize;
        }
        return *this;
    }

    __host__ void BuddyMemoryResource::cleanup() noexcept
    {
        CUresult res;
        if (!m_ctrlBlock)
            return;

        { // shared lock scope
            std::shared_lock slk{m_mtx};
            m_ctrlBlock->transactionInFlight.lock();
            if (m_ctrlBlock->refCount.fetch_sub(1, std::memory_order_seq_cst) > 1)
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
            res = cuMemUnmap(ptr, m_chunkSize);
            assert(res == ::CUDA_SUCCESS);
            res = cuMemRelease(arr[i]);
            assert(res == ::CUDA_SUCCESS);
            ptr += m_chunkSize;
        }

        // free device virtual address reservation
        res = cuMemAddressFree(m_ctrlBlock->ptr, m_maxPoolSize);
        assert(res == ::CUDA_SUCCESS);

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

    __host__ bool BuddyMemoryResource::grow()
    { // assume you already own the spinlock
        CUresult res;
        assert(m_ctrlBlock);
        if (m_ctrlBlock->size * m_chunkSize >= m_maxPoolSize)
            return false; // pool exhausted

        // check if capacity is enough
        if (m_ctrlBlock->capacity <= m_ctrlBlock->size)
        {
            void*  address = std::bit_cast<void*>(std::bit_cast<uintptr_t>(m_ctrlBlock) + sizeof(UnifiedControlBlock) +
                                                 sizeof(CUmemGenericAllocationHandle) * m_ctrlBlock->capacity);
            size_t pageSz  = static_cast<size_t>(toUnderlying(EPageSize::e4KB));
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

        res = cuMemMap(m_ctrlBlock->ptr + offset, m_chunkSize, 0, arr[m_ctrlBlock->size], 0);
        assert(res == ::CUDA_SUCCESS);
        setReadWriteDeviceVirtMemory(m_deviceId, m_ctrlBlock->ptr + offset, m_chunkSize);
        ++m_ctrlBlock->size;

        return true;
    }

    __host__ size_t BuddyMemoryResource::minOrder() const
    {
        size_t const lzcntMinSize = std::countl_zero(m_chunkSize);
        size_t const lzcntMaxSize = std::countl_zero(m_maxPoolSize);
        return lzcntMinSize - lzcntMaxSize;
    }

    __host__ size_t BuddyMemoryResource::blockToOrder(size_t size) const
    { // assumptions: size is already a POT between m_minBlockSize and m_chunkSize
        // should return: minOrder()     for size == m_chunkSize
        // should return: minOrder() + n for size == m_chunkSize >> n
        // Ensure the size is a power of two and within valid bounds
        assert(size >= m_minBlockSize && size <= m_chunkSize);
        assert((size & (size - 1)) == 0); // Check if size is a power of two

        // Calculate the minimum order
        size_t const minOrd = minOrder();

        // Calculate the order for the given size
        size_t const order = minOrd + (std::countl_zero(size) - std::countl_zero(m_chunkSize));

        return order;
    }

    __host__ size_t BuddyMemoryResource::alignUpToBlock(size_t size) const
    {
        size_t alignedSize = m_minBlockSize;
        while (alignedSize < size)
        {
            alignedSize <<= 1; // Double the size until it can accommodate 'size'.
        }
        return alignedSize;
    }

    __host__ void BuddyMemoryResource::split(size_t order, size_t nodeIndex)
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
        assert(m_ctrlBlock->allocationBitmap[leftChildIndex] == eInvalid &&
               m_ctrlBlock->allocationBitmap[rightChildIndex] == eInvalid);

        // Ensure indices are within bounds
        assert(leftChildIndex < m_ctrlBlock->allocationBitmap.size());
        assert(rightChildIndex < m_ctrlBlock->allocationBitmap.size());

        // Mark both children as free
        m_ctrlBlock->allocationBitmap[leftChildIndex]  = eFree;
        m_ctrlBlock->allocationBitmap[rightChildIndex] = eFree;
    }

    __host__ bool BuddyMemoryResource::coalesce(size_t parentIndex, size_t parentLevel)
    { // shared_lock on m_ctx, lock_guard on transactionInFlight acquired by caller.
        // parent level > min Order checked by caller
        // Ensure the parent is valid and has children
        assert(parentIndex < m_ctrlBlock->allocationBitmap.size());
        assert(m_ctrlBlock->allocationBitmap[parentIndex] == eHasChildren);

        // Calculate indices of the children
        size_t const leftChildIndex  = 2 * parentIndex + 1;
        size_t const rightChildIndex = 2 * parentIndex + 2;

        // Ensure children indices are within bounds
        assert(leftChildIndex < m_ctrlBlock->allocationBitmap.size());
        assert(rightChildIndex < m_ctrlBlock->allocationBitmap.size());

        // Check if both children are free
        if (m_ctrlBlock->allocationBitmap[leftChildIndex] == eFree && m_ctrlBlock->allocationBitmap[rightChildIndex] == eFree)
        {
            // Mark the parent as free
            m_ctrlBlock->allocationBitmap[parentIndex] = eFree;

            // Clear the children
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

    __host__ __device__ void* BuddyMemoryResource::allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align)
    {
#if !defined(__CUDA_ARCH__)
            return reinterpret_cast<BuddyMemoryResource*>(pAlloc)->allocate(sz, align);
#else
            return nullptr;
#endif
    }

    __host__ __device__ void BuddyMemoryResource::freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align)
    {
#if !defined(__CUDA_ARCH__)
            reinterpret_cast<BuddyMemoryResource*>(pAlloc)->deallocate(ptr, sz, align);
#endif
    }

    __host__ __device__ bool BuddyMemoryResource::deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID)
    {
        return reinterpret_cast<BuddyMemoryResource const*>(pAlloc)->m_deviceHnd == deviceID;
    }

    __host__ void* BuddyMemoryResource::allocate(size_t _Bytes, size_t _Align)
    {
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
            for (size_t currentLevel = level; !needsGrowth && currentLevel >= minOrder(); --currentLevel)
            {
                size_t const numNodesAtLevel = 1ULL << currentLevel;

                // Loop through nodes at the current level
                for (size_t nodeIdx = 0; !needsGrowth && nodeIdx < numNodesAtLevel; ++nodeIdx)
                {
                    size_t index = ((1ULL << currentLevel) - 1) + nodeIdx;

                    // Check if the node is free
                    if (m_ctrlBlock->allocationBitmap[index] == eFree)
                    {
                        // Split the block as needed to reach the desired level
                        while (currentLevel < level)
                        {
                            split(currentLevel, nodeIdx); // Split the parent
                            ++currentLevel;
                            nodeIdx *= 2; // Move to the left child
                        }

                        index = ((1ULL << currentLevel) - 1) + nodeIdx;
                        assert(m_ctrlBlock->allocationBitmap[index] == eFree);

                        // Compute the pointer to the allocated memory
                        size_t const offset = nodeIdx * (m_chunkSize >> (currentLevel - minOrder()));
                        CUdeviceptr  ptr    = m_ctrlBlock->ptr + offset;
                        CUdeviceptr  limit  = m_ctrlBlock->ptr + m_ctrlBlock->size * m_chunkSize;
                        if (ptr < limit)
                        {
                            // Mark the node as allocated
                            m_ctrlBlock->allocationBitmap[index] = eAllocated;
                            return std::bit_cast<void*>(ptr);
                        }
                        else
                        {
                            needsGrowth = true;
                            break;
                        }
                    }
                    else if (m_ctrlBlock->allocationBitmap[index] == eInvalid)
                        break;
                }
            }

            // If no suitable block was found, grow the pool if possible
            if (!grow())
                return nullptr;
        }
    }

    __host__ void BuddyMemoryResource::deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
        std::shared_lock slk{m_mtx};
        // Align the size to the nearest power-of-two block
        _Bytes = alignUpToBlock(_Bytes);

        // Determine the level corresponding to the block size
        size_t const level = blockToOrder(_Bytes);

        // Compute the offset of the block relative to the start of the memory pool
        size_t const offset = std::bit_cast<uintptr_t>(_Ptr) - m_ctrlBlock->ptr;

        // Calculate the index of the block in the allocation bitmap
        size_t const nodeIdx = offset / (m_chunkSize >> (level - minOrder()));
        size_t const index   = ((1ULL << level) - 1) + nodeIdx;

        // Acquire a lock for thread safety
        std::lock_guard lk{m_ctrlBlock->transactionInFlight};

        // Mark the block as free
        assert(m_ctrlBlock->allocationBitmap[index] == eAllocated);
        m_ctrlBlock->allocationBitmap[index] = eFree;

        // Attempt to coalesce blocks up the tree
        size_t parentIndex = (index - 1) / 2;
        size_t parentLevel = level - 1;
        while (parentLevel >= minOrder() && coalesce(parentIndex, parentLevel))
        {
            parentIndex = (parentIndex - 1) / 2; // Move up to the parent
            --parentLevel;
        }
    }

    __host__ __device__ bool BuddyMemoryResource::operator==(BuddyMemoryResource const& that) const noexcept
    { 
        // if control block is the same, then they should be on the same device
        assert((m_ctrlBlock == that.m_ctrlBlock && m_deviceHnd == that.m_deviceHnd) || m_deviceHnd != that.m_deviceHnd);
        return m_deviceHnd == that.m_deviceHnd && m_ctrlBlock == that.m_ctrlBlock;
    }

    // MemPoolAsyncMemoryResource -------------------------------------------------------------------------------------

    __host__ MemPoolAsyncMemoryResource::MemPoolAsyncMemoryResource(MemPoolAsyncMemoryResourceSpec const& input) :
    CudaAsyncMemoryReosurce(EMemoryResourceType::eMemPool),
    m_ctrlBlock(std::bit_cast<ControlBlock*>(input.pHostMemRes->allocate(sizeof(ControlBlock), alignof(ControlBlock)))),
    m_hostCtrlRes(input.pHostMemRes),
    m_poolSize(nextPOT(input.poolSize)),
    m_deviceId(input.deviceId)
    {
        m_host.allocateBytes = MemPoolAsyncMemoryResource::allocateBytes;
        m_host.freeBytes = MemPoolAsyncMemoryResource::freeBytes;
        m_host.allocateBytesAsync = MemPoolAsyncMemoryResource::allocateBytesAsync;
        m_host.freeBytesAsync = MemPoolAsyncMemoryResource::freeBytesAsync;
        m_host.deviceHasAccess = MemPoolAsyncMemoryResource::deviceHasAccess;
        m_host.hostHasAccess = MemPoolAsyncMemoryResource::hostHasAccess;
        initTable<<<1, 32>>>(*this);
        cudaDeviceSynchronize();

        // check successful allocation of control block
        CUresult res;
        if (m_ctrlBlock == nullptr)
        {
            if (input.pmctx)
                input.pmctx->pctx.error("Couldn't allocate control block for async memory resource, aborting...");
            std::abort();
        }
        std::construct_at(m_ctrlBlock);
        m_ctrlBlock->refCount.store(1, std::memory_order_seq_cst);

        // recover device
        CUdevice device;
        res = cuDeviceGet(&device, m_deviceId);
        assert(res == ::CUDA_SUCCESS);

        // query support for used features
        int32_t support = 0;
        res             = cuDeviceGetAttribute(&support, ::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device);
        assert(res == ::CUDA_SUCCESS);
        if (support == 0)
        {
            if (input.pmctx)
                input.pmctx->pctx.error(
                    "CUDA Memory Pools unsupported, but they are required for this allocator, aborting...");
            std::abort();
        }

        // IPC Support (not needed, here just for reference)
        // Multi-GPU support is instead queried with `cudaDeviceCanAccessPeer` and `cudaDeviceEnablePeerAccess`
        res = cuDeviceGetAttribute(&support, ::CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES, device);
        assert(res == ::CUDA_SUCCESS);
        if (support & ::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) // linux (posix) only
            if (input.pmctx)
                input.pmctx->pctx.log(
                    "device {} can create Memory Pools for Inter Process Comunication based on POSIX File Descriptors",
                    {m_deviceId});
        if (support & ::CU_MEM_HANDLE_TYPE_WIN32) // windows only
            if (input.pmctx)
                input.pmctx->pctx
                    .log("device {} can create Memory Pools for Inter Process Communication based on WIN32 Handles",
                         {m_deviceId});
        if (support & ::CU_MEM_HANDLE_TYPE_FABRIC) // CUmemFabricHandle
            if (input.pmctx)
                input.pmctx->pctx.log(
                    "device {} can create Memory Pools for Inter Process Communication based on CUmemFabricHandles",
                    {m_deviceId});
            else if (input.pmctx)
                input.pmctx->pctx.log("device {} cannot export Memory Pools to other processes", {m_deviceId});

        // create the mmeory pool with maximum threashold (meaning the CUDA Runtime won't try to free memory when
        // unoccupied until the pool is destroyed or trimmed explicitly)
        CUmemPoolProps props{};
        props.allocType     = ::CU_MEM_ALLOCATION_TYPE_PINNED; // only type supported
        props.handleTypes   = ::CU_MEM_HANDLE_TYPE_NONE;       // no IPC
        props.location.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id   = m_deviceId;
        props.maxSize       = nextPOT(input.poolSize);
        res                 = cuMemPoolCreate(&m_ctrlBlock->memPool, &props);
        assert(res == ::CUDA_SUCCESS);

        // set release threshold and reuse policies
        cuuint64_t releaseThreshold = input.releaseThreshold;
        int32_t    enabled          = 1;
        res = cuMemPoolSetAttribute(m_ctrlBlock->memPool, ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &releaseThreshold);
        assert(res == ::CUDA_SUCCESS);
        res = cuMemPoolSetAttribute(m_ctrlBlock->memPool, ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &enabled);
        assert(res == ::CUDA_SUCCESS);
        res = cuMemPoolSetAttribute(m_ctrlBlock->memPool, ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &enabled);
        assert(res == ::CUDA_SUCCESS);
        res = cuMemPoolSetAttribute(m_ctrlBlock->memPool, ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &enabled);
        assert(res == ::CUDA_SUCCESS);

        // create stream for synchronous allocations
        res = cuStreamCreate(&m_ctrlBlock->defaultStream, ::CU_STREAM_NON_BLOCKING);
        assert(res == ::CUDA_SUCCESS);
    }

    __host__ MemPoolAsyncMemoryResource::MemPoolAsyncMemoryResource(MemPoolAsyncMemoryResource const& other) :
    CudaAsyncMemoryReosurce(EMemoryResourceType::eMemPool)
    {
        std::shared_lock slk{other.m_mtx};
        m_ctrlBlock   = other.m_ctrlBlock;
        m_hostCtrlRes = other.m_hostCtrlRes;
        m_deviceId    = other.m_deviceId;
        m_poolSize    = other.m_poolSize;

        std::lock_guard lk{m_ctrlBlock->transactionInFlight};
        m_ctrlBlock->refCount.fetch_add(1, std::memory_order_seq_cst);
    }

    __host__ MemPoolAsyncMemoryResource::MemPoolAsyncMemoryResource(MemPoolAsyncMemoryResource&& other) noexcept :
    CudaAsyncMemoryReosurce(EMemoryResourceType::eMemPool)
    {
        std::lock_guard lk{other.m_mtx};
        m_ctrlBlock   = std::exchange(other.m_ctrlBlock, nullptr);
        m_hostCtrlRes = other.m_hostCtrlRes;
        m_deviceId    = other.m_deviceId;
        m_poolSize    = other.m_poolSize;
    }

    __host__ MemPoolAsyncMemoryResource& MemPoolAsyncMemoryResource::operator=(MemPoolAsyncMemoryResource const& other)
    {
        if (*this != other)
        {
            cleanup();
            std::unique_lock lk{m_mtx, std::defer_lock};
            std::shared_lock slk{other.m_mtx, std::defer_lock};
            std::lock(lk, slk);

            m_ctrlBlock   = other.m_ctrlBlock;
            m_hostCtrlRes = other.m_hostCtrlRes;
            m_deviceId    = other.m_deviceId;
            m_poolSize    = other.m_poolSize;

            std::lock_guard mlk{m_ctrlBlock->transactionInFlight};
            m_ctrlBlock->refCount.fetch_add(1, std::memory_order_seq_cst);
        }
        return *this;
    }

    __host__ MemPoolAsyncMemoryResource& MemPoolAsyncMemoryResource::operator=(MemPoolAsyncMemoryResource&& other) noexcept
    {
        if (*this != other)
        {
            cleanup();
            std::unique_lock lk{m_mtx, std::defer_lock};
            std::unique_lock slk{other.m_mtx, std::defer_lock};
            std::lock(lk, slk);

            m_ctrlBlock   = std::exchange(other.m_ctrlBlock, nullptr);
            m_hostCtrlRes = other.m_hostCtrlRes;
            m_deviceId    = other.m_deviceId;
            m_poolSize    = other.m_poolSize;
        }
        return *this;
    }

    __host__ MemPoolAsyncMemoryResource::~MemPoolAsyncMemoryResource() noexcept
    {
        cleanup();
    }

    __host__ void* MemPoolAsyncMemoryResource::allocate(size_t _Bytes, size_t _Align)
    {
        CUresult         res;
        std::shared_lock slk{m_mtx};                           // lock to access object
        std::lock_guard  lk{m_ctrlBlock->transactionInFlight}; // lock to allocate memory
        void*            ret = performAlloc(m_ctrlBlock->defaultStream, _Bytes);
        res                  = cuStreamSynchronize(m_ctrlBlock->defaultStream);
        assert(res == ::CUDA_SUCCESS);
        return ret;
    }

    __host__ void MemPoolAsyncMemoryResource::deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
        CUresult         res;
        std::shared_lock slk{m_mtx};                           // lock to access object
        std::lock_guard  lk{m_ctrlBlock->transactionInFlight}; // lock to allocate memory
        res = cuMemFreeAsync(std::bit_cast<CUdeviceptr>(_Ptr), m_ctrlBlock->defaultStream);
        assert(res == ::CUDA_SUCCESS);
        res = cuStreamSynchronize(m_ctrlBlock->defaultStream);
        assert(res == ::CUDA_SUCCESS);
    }

    __host__ __device__ bool MemPoolAsyncMemoryResource::operator==(MemPoolAsyncMemoryResource const& other) const noexcept
    { // Doesn't work if Multi GPU support is introduced
        bool sameCtrlBlock = other.m_ctrlBlock == m_ctrlBlock;
        bool sameDevice    = m_deviceId == other.m_deviceId;
        assert((sameCtrlBlock && sameDevice) || !sameCtrlBlock);
        return sameDevice && sameCtrlBlock;
    }

    __host__ void* MemPoolAsyncMemoryResource::allocate_async(size_t sz, size_t align, cuda::stream_ref streamRef)
    {
        std::shared_lock slk{m_mtx};                           // lock to access object
        std::lock_guard  lk{m_ctrlBlock->transactionInFlight}; // lock to allocate memory
        CUstream         stream = static_cast<CUstream>(streamRef.get());
        return performAlloc(stream, sz);
    }

    __host__ void* MemPoolAsyncMemoryResource::performAlloc(CUstream stream, size_t sz)
    {
        CUdeviceptr ptr = 0;
        CUresult    res = cuMemAllocFromPoolAsync(&ptr, sz, m_ctrlBlock->memPool, stream);
        if (res == ::CUDA_SUCCESS)
            return std::bit_cast<void*>(ptr);
        else if (res == ::CUDA_ERROR_OUT_OF_MEMORY)
            return nullptr;
        else
        {
            assert(false);
            return nullptr;
        }
    }

    __host__ __device__ void* MemPoolAsyncMemoryResource::allocateBytes(BaseMemoryResource* pAlloc, size_t sz, size_t align)
    {
#if defined(__CUDA_ARCH__)
        assert(false);
        return nullptr;
#else
        return reinterpret_cast<MemPoolAsyncMemoryResource*>(pAlloc)->allocate(sz, align);
#endif
    }

    __host__ __device__ void MemPoolAsyncMemoryResource::freeBytes(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align)
    {
#if defined(__CUDA_ARCH__)
        assert(false);
#else
        reinterpret_cast<MemPoolAsyncMemoryResource*>(pAlloc)->deallocate(ptr, sz, align);
#endif
    }

    __host__ void* MemPoolAsyncMemoryResource::allocateBytesAsync(BaseMemoryResource* pAlloc, size_t sz, size_t align, CudaStreamHandle stream)
    {
        return reinterpret_cast<MemPoolAsyncMemoryResource*>(pAlloc)->allocate_async(sz, align, streamRefFromHandle(stream));
    }

    __host__ void MemPoolAsyncMemoryResource::freeBytesAsync(BaseMemoryResource* pAlloc, void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        reinterpret_cast<MemPoolAsyncMemoryResource*>(pAlloc)->deallocate_async(ptr, sz, align, streamRefFromHandle(stream));
    }

    __host__ __device__ bool MemPoolAsyncMemoryResource::deviceHasAccess(BaseMemoryResource const* pAlloc, int32_t deviceID)
    {
		return reinterpret_cast<MemPoolAsyncMemoryResource const*>(pAlloc)->m_deviceId == deviceID;
    }

    __host__ void MemPoolAsyncMemoryResource::deallocate_async(void* ptr, size_t sz, size_t align, cuda::stream_ref streamRef)
    {                                                          // should be called on the same stream as the allocation
        std::shared_lock slk{m_mtx};                           // lock to access object
        std::lock_guard  lk{m_ctrlBlock->transactionInFlight}; // lock to allocate memory
        CUstream         stream = static_cast<CUstream>(streamRef.get());
        CUresult         res    = cuMemFreeAsync(std::bit_cast<CUdeviceptr>(ptr), stream);
        assert(res == ::CUDA_SUCCESS);
    }

    __host__ void MemPoolAsyncMemoryResource::cleanup() noexcept
    {
        CUresult res;
        if (!m_ctrlBlock)
            return;

        { // shared lock scope (decrement ref counter)
            std::shared_lock slk{m_mtx};
            m_ctrlBlock->transactionInFlight.lock();
            if (m_ctrlBlock->refCount.fetch_sub(1, std::memory_order_seq_cst) > 1)
            {
                m_ctrlBlock->transactionInFlight.unlock();
                return;
            }
        }
        std::lock_guard lk{m_mtx};
        if (!m_ctrlBlock)
            return;

        // destroy memory pool
        cuCtxSynchronize();
        res = cuMemPoolDestroy(m_ctrlBlock->memPool);
        assert(res == ::CUDA_SUCCESS);

        // destroy default stream
        res = cuStreamDestroy(m_ctrlBlock->defaultStream);
        assert(res == ::CUDA_SUCCESS);

        // trigger destructor of control block
        std::destroy_at(m_ctrlBlock);
        m_hostCtrlRes->deallocate(m_ctrlBlock, sizeof(ControlBlock), alignof(ControlBlock));
        m_ctrlBlock = nullptr;
    }

    // Memory Resouce Boilerplate -------------------------------------------------------------------------------------
    __host__ __device__ void switchOnMemoryResource(
        EMemoryResourceType eAlloc,
        BaseMemoryResource* p,
        size_t*             sz,
        bool                destroy,
        void*               ctorParam)
    {
        EMemoryResourceType category = extractCategory(eAlloc);
        EMemoryResourceType type     = extractType(eAlloc);
        switch (category)
        {
            using enum EMemoryResourceType;
            case eHost:
                switch (type)
                {
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
                        break;
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
                        break;
                    case eMemPool:
                        if (p)
                            if (destroy)
                                std::destroy_at(std::bit_cast<MemPoolAsyncMemoryResource*>(p));
                            else
                                std::construct_at(std::bit_cast<MemPoolAsyncMemoryResource*>(p),
                                                  *std::bit_cast<MemPoolAsyncMemoryResourceSpec*>(ctorParam));
                        else if (sz)
                            *sz = sizeof(MemPoolAsyncMemoryResource);
                        break;
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

    __host__ __device__ size_t sizeForMemoryResource(EMemoryResourceType eAlloc)
    {
        size_t ret = 0;
        switchOnMemoryResource(eAlloc, nullptr, &ret, true, nullptr);
        return ret;
    }

    __host__ BaseMemoryResource* constructMemoryResourceAt(void* ptr, EMemoryResourceType eAlloc, void* ctorParam)
    {
        BaseMemoryResource* p = std::bit_cast<BaseMemoryResource*>(ptr);
        switchOnMemoryResource(eAlloc, p, nullptr, false, ctorParam);
        return p;
    }

    __host__ void destroyMemoryResourceAt(BaseMemoryResource* p, EMemoryResourceType eAlloc)
    {
        switchOnMemoryResource(eAlloc, p, nullptr, true, nullptr);
    }

    __host__ __device__ EMemoryResourceType categoryOf(BaseMemoryResource* allocator)
    {
        return extractCategory(allocator->type);
    }

    __host__ __device__ bool isDeviceAllocator(BaseMemoryResource* allocator, int32_t deviceId)
    {
        return allocator->deviceHasAccess(deviceId);
    }

    __host__ __device__ bool isHostAllocator(BaseMemoryResource* allocator) { return allocator->hostHasAccess(); }

    // BaseDeviceContainer --------------------------------------------------------------------------------------------
    // TODO: use cuda::atomic_ref instead of atomic primitives. Reference: https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html
    __host__ __device__ void BaseDeviceContainer::lockForRead() const
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

    __host__ __device__ void BaseDeviceContainer::unlockForRead() const
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

    __host__ __device__ bool BaseDeviceContainer::lockForWrite() const
    {
        int expected = 0;
#if defined(__CUDA_ARCH__)
        // bacause of warp execution (cc < 7.0), the atomic exchange to 1 cannot possibly go right within a warp
        // therefore, *only the "warp leader"* (warpIndex == 0) will lock the thing, and should perform the mutating operations
        // Note: This will cause divergence. once you unlock, you should `__syncthreads` (or `__syncwarp` if cc >= 7.9)
        int32_t warpIndex = warpWideThreadIndex();
        if (warpIndex == leaderWarpIndex)
        {
            while (atomicAdd(&m_readCount, 0) != 0 || atomicCAS(&m_writeCount, expected, 1) != expected)
            {
            }
            return true;
        }
        else
            return false;
#else
        std::atomic_ref<int> writeRef(m_writeCount);
        std::atomic_ref<int> readRef(m_readCount);
        // Wait until no reader or writer is active
        while (readRef.load(std::memory_order_acquire) > 0 ||
               !writeRef.compare_exchange_strong(expected, 1, std::memory_order_seq_cst, std::memory_order_seq_cst))
        {
            // Spin-wait
        }
        return true;
#endif
    }

    __host__ __device__ void BaseDeviceContainer::unlockForWrite() const
    {
#if defined(__CUDA_ARCH__)
        // Decrement writer count
        int32_t warpIndex = warpWideThreadIndex();
        if (warpIndex == leaderWarpIndex)
        {
            atomicExch(&m_writeCount, 0);
        }
#else
        std::atomic_ref<int> writeRef(m_writeCount);
        // Decrement writer count
        writeRef.fetch_sub(1, std::memory_order_release);
#endif
    }

    __host__ __device__ void BaseDeviceContainer::waitWriter() const
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
    __host__ __device__ DynaArray::DynaArray(DynaArray const& other) :
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

    __host__ __device__ DynaArray::DynaArray(DynaArray&& other) noexcept :
    BaseDeviceContainer(other.m_resource, other.stream)
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

    __host__ __device__ void DynaArray::reserve(size_t newCapacity, bool lock)
    {
        if (lock)
            lockForWrite();
        if (newCapacity <= m_capacity)
        {
            if (lock)
                unlockForWrite();
            return;
        }

        void* newHead = m_resource->tryAllocateAsync(newCapacity * m_elemSize, alignof(std::max_align_t), stream);
        if (m_head)
        {
            std::memcpy(newHead, m_head, m_size * m_elemSize);
            m_resource->tryFreeAsync(m_head, m_size * m_elemSize, alignof(std::max_align_t), stream);
        }
        m_head     = newHead;
        m_capacity = newCapacity;
        if (lock)
            unlockForWrite();
    }

    __host__ __device__ void DynaArray::clear(bool lock) noexcept
    {
        if (lock)
            lockForWrite();
        if (m_head)
        {
            m_resource->tryFreeAsync(m_head, m_size * m_elemSize, alignof(std::max_align_t), stream);
            m_head = nullptr;
        }
        if (lock)
            unlockForWrite();
    }

    __host__ __device__ bool DynaArray::push_back(void const* pValue, bool srcHost, bool lock)
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
        int32_t device;
        auto    cudaret = cudaGetDevice(&device);
        assert(cudaret == ::cudaSuccess);
        cudaMemcpyKind kind = isDeviceAllocator(m_resource, device)
                                  ? (srcHost ? ::cudaMemcpyHostToDevice : ::cudaMemcpyDeviceToDevice)
                                  : (srcHost ? ::cudaMemcpyHostToHost : ::cudaMemcpyDeviceToHost);
        cudaret             = cudaMemcpy(dest, pValue, m_elemSize, kind);
        if (cudaret != ::cudaSuccess)
            ret = false;
#endif
        if (ret)
            ++m_size;

        if (lock)
            unlockForWrite();
        return ret;
    }

    __host__ __device__ void DynaArray::pop_back(bool lock)
    {
        if (lock)
            lockForWrite();

        if (m_size != 0)
            --m_size;

        if (lock)
            unlockForWrite();
    }

    // assumes you already locked for read

    __host__ __device__ void* DynaArray::at(size_t index)
    {
        if (eligibleForAccess(index))
            return std::bit_cast<void*>(std::bit_cast<uintptr_t>(m_head) + index * m_elemSize);
        else
            return nullptr;
    }

    __host__ __device__ void const* DynaArray::atConst(size_t index) const
    {
        if (eligibleForAccess(index))
            return std::bit_cast<void const*>(std::bit_cast<uintptr_t>(m_head) + index * m_elemSize);
        else
            return nullptr;
    }

    __host__ __device__ void DynaArray::copyFrom(DynaArray const& other)
    {
        if (other.m_size > 0)
        {
            std::memcpy(m_head, other.m_head, other.m_size * other.m_elemSize);
            m_size = other.m_size;
        }
    }

    __host__ __device__ bool DynaArray::eligibleForAccess(size_t index) const
    {
        assert(index < m_size);
#if defined(__CUDA_ARCH__)
        cudaError_t cudaRes;
        int32_t     device;
        cudaRes = cudaGetDevice(&device);
        assert(cudaRes == ::cudaSuccess);
        if (isDeviceAllocator(m_resource, device))
            return true;
        else
            return false;
#else
        if (isHostAllocator(m_resource))
            return true;
        else
        { // call cudaMemcpy? no, wasteful
            assert(false);
            return false;
        }
#endif
    }

    __host__ __device__ DynaArray& DynaArray::operator=(DynaArray const& other)
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

    __host__ __device__ DynaArray& DynaArray::operator=(DynaArray&& other) noexcept
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

    __host__ __device__ DynaArray::~DynaArray() noexcept { clear(); }

    __host__ bool DynaArray::copyToHostSync(void* /*DMT_RESTRICT*/ dest, bool lock) const
    {
        cudaError_t err;
        int32_t     device;
        err = cudaGetDevice(&device);
        assert(err == ::cudaSuccess);
        if (!isDeviceAllocator(m_resource, device))
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

    __host__ __device__ size_t DynaArray::size(bool lock) const
    {
        size_t ret = 0;
        if (lock)
            lockForRead();
        ret = m_size;
        if (lock)
            unlockForRead();
        return ret;
    }

    __host__ __device__ size_t DynaArray::capacity(bool lock) const
    {
        size_t ret = 0;
        if (lock)
            lockForRead();
        ret = m_capacity;
        if (lock)
            unlockForRead();
        return ret;
    }
} // namespace dmt
