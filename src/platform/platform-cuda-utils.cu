#include "platform-cuda-utils.cuh"

#define DMT_INTERFACE_AS_HEADER
#include "platform-cuda-utils.h"
#include "platform-memory.h"

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#include <bit>

#include <cstdint>
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

    // Memory Resouce Implementations ---------------------------------------------------------------------------------
    void* HostPoolReousce::allocateBytes(size_t sz, size_t align) { return m_res.allocate(sz, align); }
    void  HostPoolReousce::freeBytes(void* ptr, size_t sz, size_t align) { return m_res.deallocate(ptr, sz, align); }
    void* HostPoolReousce::allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream)
    {
        assert(false);
        return nullptr;
    }

    void HostPoolReousce::freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) { assert(false); }

    void* HostPoolReousce::do_allocate(size_t _Bytes, size_t _Align) { return m_res.allocate(_Bytes, _Align); }
    void  HostPoolReousce::do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align)
    {
        m_res.deallocate(_Ptr, _Bytes, _Align);
    }
    bool HostPoolReousce::do_is_equal(memory_resource const& _That) const noexcept { return m_res == _That; }

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

    // Memory Resouce Boilerplate -------------------------------------------------------------------------------------
    void switchOnMemoryResoure(EMemoryResourceType eAlloc, BaseMemoryResource* p, size_t* sz, bool destroy)
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
                                std::destroy_at(std::bit_cast<HostPoolReousce*>(p));
                            else
                                std::construct_at(std::bit_cast<HostPoolReousce*>(p));
                        else if (sz)
                            *sz = sizeof(HostPoolReousce);
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

    size_t sizeForMemoryResouce(EMemoryResourceType eAlloc)
    {
        size_t ret = 0;
        switchOnMemoryResoure(eAlloc, nullptr, &ret, true);
        return ret;
    }

    BaseMemoryResource* constructMemoryResourceAt(void* ptr, EMemoryResourceType eAlloc)
    {
        BaseMemoryResource* p = std::bit_cast<BaseMemoryResource*>(ptr);
        switchOnMemoryResoure(eAlloc, p, nullptr, false);
        return p;
    }

    void destroyMemoryResouceAt(BaseMemoryResource* p, EMemoryResourceType eAlloc)
    {
        switchOnMemoryResoure(eAlloc, p, nullptr, true);
    }

    EMemoryResourceType categoryOf(BaseMemoryResource* allocator)
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

    void* allocateFromCategory(BaseMemoryResource* allocator, size_t sz, size_t align, CudaStreamHandle stream)
    {
        if (auto* a = dynamic_cast<CudaAsyncMemoryReosurce*>(allocator); a)
        {
            if (stream != noStream)
            {
                cuda::stream_ref streamref = streamRefFromHandle(stream);
                return a->allocate_async(sz, align, streamref);
            }
            else
                return a->allocate(sz, align);
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

    void freeFromCategory(BaseMemoryResource* allocator, void* ptr, size_t sz, size_t align, CudaStreamHandle stream)
    {
        if (auto* a = dynamic_cast<CudaAsyncMemoryReosurce*>(allocator); a)
        {
            if (stream != noStream)
            {
                cuda::stream_ref streamref = streamRefFromHandle(stream);
                a->deallocate_async(ptr, sz, align, streamref);
            }
            else
                a->deallocate(ptr, sz, align);
        }
        else if (auto* a = dynamic_cast<DeviceMemoryReosurce*>(allocator); a)
            a->deallocate(ptr, sz, align);
        else if (auto* a = dynamic_cast<UnifiedMemoryResource*>(allocator); a)
            a->deallocate(ptr, sz, align);
        else if (auto* a = dynamic_cast<std::pmr::memory_resource*>(allocator); a)
            a->deallocate(ptr, sz, align);
    }

    // BlockyForwardList ----------------------------------------------------------------------------------------------
    DMT_CPU_GPU void* BlockyForwardList::allocateNode()
    {
        return allocateFromCategory(m_resource, nodeSize, alignof(NodeHeader), stream);
    }

    DMT_CPU_GPU void BlockyForwardList::freeNode(void* node)
    {
        freeFromCategory(m_resource, node, nodeSize, alignof(NodeHeader), stream);
    }

    DMT_CPU_GPU void BlockyForwardList::lockForRead()
    {
#if defined(__CUDA_ARCH__)
    // Wait until no writer is active
    while (atomicAdd(&m_writeCount, 0) > 0) {
        // Spin-wait
    }
    // Increment reader count
    atomicAdd(&m_readCount, 1);
#else
    std::atomic_ref<int> writeRef(m_writeCount);
    std::atomic_ref<int> readRef(m_readCount);
    // Wait until no writer is active
    while (writeRef.load(std::memory_order_acquire) > 0) {
        // Spin-wait
    }
    // Increment reader count
    readRef.fetch_add(1, std::memory_order_acquire);
#endif    
    }

    DMT_CPU_GPU void BlockyForwardList::unlockForRead()
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

    DMT_CPU_GPU void BlockyForwardList::lockForWrite()
    {
#if defined(__CUDA_ARCH__)
    // Wait until no reader or writer is active
    while (atomicAdd(&m_readCount, 0) > 0 || atomicAdd(&m_writeCount, 0) > 0) {
        // Spin-wait
    }
    // Increment writer count
    atomicAdd(&m_writeCount, 1);
#else
    std::atomic_ref<int> writeRef(m_writeCount);
    std::atomic_ref<int> readRef(m_readCount);
    // Wait until no reader or writer is active
    while (readRef.load(std::memory_order_acquire) > 0 || writeRef.load(std::memory_order_acquire) > 0) {
        // Spin-wait
    }
    // Increment writer count
    writeRef.fetch_add(1, std::memory_order_acquire);
#endif
    }

    DMT_CPU_GPU void BlockyForwardList::unlockForWrite()
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
} // namespace dmt