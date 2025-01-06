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
    DMT_CPU_GPU static constexpr cuda::stream_ref streamRefFromHandle(CudaStreamHandle handle)
    {
        return {std::bit_cast<::cudaStream_t>(handle)};
    }

    // Memory Resouce Interfaces --------------------------------------------------------------------------------------
    // cannot derive std::pmr::memory_resouce here cause we need __device__ on the allocate
    class DMT_INTERFACE DeviceMemoryReosurce : public BesaMemoryResource
    {
    public:
        // BesaMemoryResouce
        void* allocateBytes(size_t sz, size_t align) override { return do_allocate(sz, align); }
        void  freeBytes(void* ptr, size_t sz, size_t align) override { do_deallocate(ptr, sz, align); }
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override
        {
            assert(false);
            return nullptr;
        }
        void freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override { assert(false); }

        DMT_CPU_GPU void* allocate(size_t sz, size_t align) { return do_allocate(sz, align); }
        DMT_CPU_GPU void  deallocate(void* ptr, size_t sz, size_t align) { do_deallocate(ptr, sz, align); }
        DMT_CPU_GPU bool  operator==(DeviceMemoryReosurce const&) const noexcept { return true; }

    private:
        DMT_CPU_GPU virtual void* do_allocate(size_t sz, size_t align)              = 0;
        DMT_CPU_GPU virtual void  do_deallocate(void* ptr, size_t sz, size_t align) = 0;
    };
    static_assert(cuda::mr::resource<DeviceMemoryReosurce>);

    class DMT_INTERFACE CudaAsyncMemoryReosurce : public BesaMemoryResource, public std::pmr::memory_resource
    {
    public:
        void* allocateBytes(size_t sz, size_t align) override { return allocate(sz, align); }
        void  freeBytes(void* ptr, size_t sz, size_t align) override { return deallocate(ptr, sz, align); }
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override
        {
            assert(isValidHandle(stream));
            return do_allocate_async(sz, align, streamRefFromHandle(stream));
        }
        void freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override
        {
            assert(isValidHandle(stream));
            do_deallocate_async(ptr, sz, align, streamRefFromHandle(stream));
        }

        DMT_CPU void* allocate_async(size_t sz, size_t align, cuda::stream_ref stream)
        {
            assert((align & (align - 1)) == 0 && "alignment should be a power of two");
            return do_allocate_async(sz, align, stream);
        }
        DMT_CPU void deallocate_async(void* ptr, size_t sz, size_t align, cuda::stream_ref stream)
        {
            assert((align & (align - 1)) == 0 && "alignment should be a power of two");
            do_deallocate_async(ptr, sz, align, stream);
        }

    private:
        DMT_CPU virtual void* do_allocate_async(size_t, size_t, cuda::stream_ref)          = 0;
        DMT_CPU virtual void  do_deallocate_async(void*, size_t, size_t, cuda::stream_ref) = 0;
    };
    static_assert(cuda::mr::async_resource<CudaAsyncMemoryReosurce>);

    // Memory Resouce Implementations ---------------------------------------------------------------------------------
    class HostPoolReousce : public DeviceMemoryReosurce, public std::pmr::memory_resource
    {
    private:
        void* allocateBytes(size_t sz, size_t align) override { return m_res.allocate(sz, align); }
        void  freeBytes(void* ptr, size_t sz, size_t align) override { return m_res.deallocate(ptr, sz, align); }
        void* allocatesBytesAsync(size_t sz, size_t align, CudaStreamHandle stream) override
        {
            assert(false);
            return nullptr;
        }

        void freeBytesAsync(void* ptr, size_t sz, size_t align, CudaStreamHandle stream) override { assert(false); }

        void* do_allocate(size_t _Bytes, size_t _Align) override { return m_res.allocate(_Bytes, _Align); }
        void  do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override
        {
            m_res.deallocate(_Ptr, _Bytes, _Align);
        }
        bool do_is_equal(memory_resource const& _That) const noexcept override { return m_res == _That; }

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
        DMT_CPU_GPU void* do_allocate(size_t sz, [[maybe_unused]] size_t align) override
        {
            void* tmp = nullptr;
            if (cudaMalloc(&tmp, sz) != ::cudaSuccess)
                return nullptr;
            return tmp;
        }

        DMT_CPU_GPU void do_deallocate(void* ptr, size_t sz, size_t align) override
        {
            assert(cudaFree(ptr) == ::cudaSuccess);
        }
    };

    class CudaMallocAsyncResource : public CudaAsyncMemoryReosurce
    {
    private:
        void* do_allocate(size_t _Bytes, [[maybe_unused]] size_t _Align) override
        {
            void* tmp = nullptr;
            if (cudaMalloc(&tmp, _Bytes) != ::cudaSuccess)
                return nullptr;
            return tmp;
        }
        void do_deallocate(void* _Ptr, size_t _Bytes, [[maybe_unused]] size_t _Align) override
        {
            assert(cudaFree(_Ptr) == ::cudaSuccess);
        }
        DMT_CPU void* do_allocate_async(size_t sz, [[maybe_unused]] size_t align, cuda::stream_ref stream) override
        {
            void* tmp = nullptr;
            if (cudaMallocAsync(&tmp, sz, stream.get()) != ::cudaSuccess)
                return nullptr;
            return tmp;
        }
        DMT_CPU void do_deallocate_async(void*                   ptr,
                                         [[maybe_unused]] size_t sz,
                                         [[maybe_unused]] size_t align,
                                         cuda::stream_ref        stream) override
        {
            assert(cudaFreeAsync(ptr, stream.get()) == ::cudaSuccess);
        }
        bool do_is_equal(memory_resource const& _That) const noexcept override { return true; }
    };

    // Memory Resouce Boilerplate -------------------------------------------------------------------------------------
    static void switchOnMemoryResoure(EMemoryResourceType eAlloc, BesaMemoryResource* p, size_t* sz, bool destroy)
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

    BesaMemoryResource* constructMemoryResourceAt(void* ptr, EMemoryResourceType eAlloc)
    {
        BesaMemoryResource* p = std::bit_cast<BesaMemoryResource*>(ptr);
        switchOnMemoryResoure(eAlloc, p, nullptr, false);
        return p;
    }

    void destroyMemoryResouceAt(BesaMemoryResource* p, EMemoryResourceType eAlloc)
    {
        switchOnMemoryResoure(eAlloc, p, nullptr, true);
    }
} // namespace dmt