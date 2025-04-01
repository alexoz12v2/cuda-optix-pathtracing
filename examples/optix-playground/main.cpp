#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "cuda-wrappers/cuda-nvrtc.h"

#include <optix.h>
// maybe to remove
#include <optix_stubs.h>

// TODO add an overload of cuda driver cal which accepts the name of the function

// must be defined in only one translation unit if stubs is included
OptixFunctionTable g_optixFunctionTable = {};

namespace dmt {
    // TODO 2 versions: one with library loaded, hence can get you the error string, another without it
    bool optixCall(OptixResult result)
    {
        if (result != ::OPTIX_SUCCESS)
        {
            Context ctx;
            if (ctx.isValid())
                ctx.error("OptiX Failed", {});
            return false;
        }

        return true;
    }
} // namespace dmt

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor()
        {
            if (cudaApi->m_library)
            {
                if (optixContext)
                {
                    optixDeviceContextDestroy(optixContext);
                }

                if (cuCtx)
                    cudaApi->cuCtxDestroy(cuCtx);
            }

            if (nvrtcApi->m_library)
                loader.unloadLibrary(nvrtcApi->m_library);
            if (cudaApi->m_library)
                loader.unloadLibrary(cudaApi->m_library);

            dmt::Ctx::destroy();
        }

        // OptiX resoruces
        OptixDeviceContext optixContext = nullptr;

        // CUDA resoruces
        CUcontext cuCtx = 0;

        // libraries
        dmt::os::LibraryLoader                         loader{false};
        std::unique_ptr<NvcudaLibraryFunctions>        cudaApi  = std::make_unique<NvcudaLibraryFunctions>();
        std::unique_ptr<Nvrtc64_120_0LibraryFunctions> nvrtcApi = std::make_unique<Nvrtc64_120_0LibraryFunctions>();
    } j;

    // `dmt::Ctx::destroy` requires all Contextes to be destroyed
    {
        dmt::Context ctx;
        ctx.log("Hello World!", {});

        // 0. load CUDA driver api and NVRTC library
        if (!loadNvcudaFunctions(j.loader, j.cudaApi.get()))
        {
            ctx.error("Couldn't load CUDA Driver APi", {});
            return 1;
        }
        dmt::fixCUDADriverSymbols(j.cudaApi.get());

        if (!loadNvrtc64_120_0Functions(j.loader, j.nvrtcApi.get()))
        {
            ctx.error("Couldn't load NVRTC library", {});
            return 1;
        }

        // 1. Create CUDA Driver Context
        CUdevice device = -1;
        j.cudaApi->cuInit(0);
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuDeviceGet(&device, 0)))
            return 1;
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxCreate(&j.cuCtx, 0, device)))
            return 1;

        // 2. Initialize OptiX Library
        if (!dmt::optixCall(optixInit()))
            return 1;

        // 3. Create OptiX Context
        // clang-format off
        OptixDeviceContextOptions const options{
            .logCallbackFunction = [](unsigned int level, char const* tag, char const* message, void* cbdata) {
                dmt::Context ctx;
                if (ctx.isValid())
                {
                    size_t const len = strlen(tag);
                    if (level <= 2)
                        ctx.error("{}: {}", std::make_tuple(std::string_view(tag, len), std::string_view(message)));
                    if (level == 3)
                        ctx.warn("{}: {}", std::make_tuple(std::string_view(tag, len), std::string_view(message)));
                    else
                        ctx.log("{}: {}", std::make_tuple(std::string_view(tag, len), std::string_view(message)));
                }
            },
            .logCallbackData = nullptr,
            .logCallbackLevel = 4,
            .validationMode = ::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
        };
        // clang-format on

        // Disable caching jsut for the moment
        dmt::os::env::set("OPTIX_CACHE_MAXSIZE", "0");
        if (!dmt::optixCall(optixDeviceContextCreate(j.cuCtx, &options, &j.optixContext)))
            return 1;

        ctx.log("OptiX Context Created", {});
    }

    return 0;
}