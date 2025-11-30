#include "cuda-wrappers-utils.h"

#include "platform/platform-context.h"

namespace dmt {
    bool cudaDriverCall(CUDADriverLibrary const* cudaApi, CUresult result)
    {
        if (result == ::CUDA_SUCCESS)
            return true;

        Context     ctx;
        char const* errorStr = nullptr;
        cudaApi->cuGetErrorString(result, &errorStr);

        if (errorStr != nullptr)
            ctx.error("CUDA Driver Error: {}", std::make_tuple(std::string_view(errorStr)));
        else
            ctx.error("CUDA Driver Unrecognized Error (did you forget to call cuInit?)", {});

        return false;
    }

    void fixCUDADriverSymbols(CUDADriverLibrary* cudaApi)
    {
#if defined(DMT_OS_WINDOWS)
        // 1. apparently, as of CUDA 12.6, driver version ~540, cuCtxCreate_v4 is the latest creation
        // function, but we have a destroy up to v2
        cudaApi->cuCtxCreate = reinterpret_cast<decltype(cudaApi->cuCtxCreate)>(
            dmt::os::lib::getFunc(cudaApi->m_library, "cuCtxCreate_v2"));
        assert(cudaApi->cuCtxCreate);
        cudaApi->cuCtxDestroy = reinterpret_cast<decltype(cudaApi->cuCtxDestroy)>(
            dmt::os::lib::getFunc(cudaApi->m_library, "cuCtxDestroy_v2"));
        assert(cudaApi->cuCtxDestroy);
#endif
    }

} // namespace dmt