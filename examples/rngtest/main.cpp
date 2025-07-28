#define DMT_ENTRY_POINT
#include "platform/platform.h"

int guardedMain()
{
    dmt::Ctx::init();
    struct Janitor
    {
        ~Janitor()
        {
            if (cudaApi && cuCtx)
            {
                cudaApi->cuCtxDestroy(cuCtx);
            }

            if (nvrtcApi && nvrtcApi->m_library)
                loader.unloadLibrary(nvrtcApi->m_library);
            if (cudaApi && cudaApi->m_library)
                loader.unloadLibrary(cudaApi->m_library);

            dmt::Ctx::destroy();
        }

        CUcontext                                      cuCtx = 0;
        dmt::os::LibraryLoader                         loader{false};
        std::unique_ptr<NvcudaLibraryFunctions>        cudaApi  = nullptr;
        std::unique_ptr<Nvrtc64_120_0LibraryFunctions> nvrtcApi = nullptr;
    } j;

    j.cudaApi  = std::make_unique<NvcudaLibraryFunctions>();
    j.nvrtcApi = std::make_unique<Nvrtc64_120_0LibraryFunctions>();

    // when dmt::Ctx::destroy runs, no active Context instances are allowed on the same thread, otherwise deadlock
    {
        dmt::Context ctx;
        if (!ctx.isValid())
            return 1;

        if (!loadNvcudaFunctions(j.loader, j.cudaApi.get()))
        {
            ctx.error("Couldn't load nvcuda.dll", {});
            return 1;
        }
        dmt::fixCUDADriverSymbols(j.cudaApi.get());
        j.cudaApi->cuInit(0);
        CUdevice device = -1;
        j.cudaApi->cuDeviceGet(&device, 0);
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxCreate(&j.cuCtx, 0, device)))
            return 1;

        if (!loadNvrtc64_120_0Functions(j.loader, j.nvrtcApi.get()))
        {
            ctx.error("Couldn't load CUDA NVRTC Library", {});
            return 1;
        }

        //todo: insert the cuda rng
    }

    return 0;
}