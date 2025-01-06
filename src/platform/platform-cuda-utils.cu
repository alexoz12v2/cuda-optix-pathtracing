#define DMT_INTERFACE_AS_HEADER
#include "platform-cuda-utils.h"
#include "platform-memory.h"

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
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
        mctx->pctx.log("Chosed Device: {}", {actualProps.name});
        mctx->pctx.log("Compute Capability: {}.{}", {actualProps.major, actualProps.minor});
        assert(actualProps.managedMemory && actualProps.canMapHostMemory);

        ret.warpSize = actualProps.warpSize;

        // forrce CUDA context initialization
        if (cudaFree(nullptr) != ::cudaSuccess)
        {
            mctx->pctx.error("Couldn't initialize CUDA context");
            ret.cudaCapable = false;
            return ret;
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
} // namespace dmt