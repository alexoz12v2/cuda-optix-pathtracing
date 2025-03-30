#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "application/application-display.h"

static void testDisplay(NvcudaLibraryFunctions* cudaApi, Nvrtc64_120_0LibraryFunctions* nvrtcApi)
{
    dmt::Context           ctx;
    dmt::Display           displayGUI;
    dmt::CUDASurfaceDrawer drawer{cudaApi, nvrtcApi};
    if (drawer.isValid())
        displayGUI.ShowWindow(&drawer);
    else
        ctx.error("Error Creating Context. Cannot instanciate window", {});
}

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

        testDisplay(j.cudaApi.get(), j.nvrtcApi.get());
    }

    return 0;
}