#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"

#include "core/core-parser.h"
#include "core/core-render.h"

#include "cuda-wrappers/cuda-wrappers-cuda-driver.h"
#include "cuda-wrappers/cuda-wrappers-utils.h"
#include "cuda-wrappers/cuda-wrappers-nvrtc.h"

namespace /*static*/ {
    using namespace dmt;

    static char const* s_saxpySrc = R"a(
template <typename T>
extern "C" __global__ void saxpy_grid_stride(int n, T a, const T* x, T* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}
)a";

    static std::vector<char const*> const s_nvccOpts{
        "-std=c++20",
        "-arch",
        "compute_60",
        "-lineinfo",
        "-G",
        "--use_fast_math",
        "-default-device",
        "-rdc",
        "true",
    };
} // namespace

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor()
        {
            if (cuCtx)
                cudaApi->cuCtxDestroy(cuCtx);

            if (m_nvrtcLoaded)
                loader.unloadLibrary(nvrtcApi->m_library);
            if (m_cudaLoaded)
                loader.unloadLibrary(cudaApi->m_library);
            dmt::Ctx::destroy();
        }

        bool loadCUDA()
        {
            if (!cudaApi)
                return false;

            m_cudaLoaded = loadCUDADriverLibraryFunctions(loader, cudaApi.get());
            if (m_cudaLoaded)
            {
                dmt::fixCUDADriverSymbols(cudaApi.get());
            }

            return m_cudaLoaded;
        }

        bool loadNVRTC()
        {
            if (!nvrtcApi)
                return false;
            m_nvrtcLoaded = loadNVRTCLibraryFunctions(loader, nvrtcApi.get());
            return m_nvrtcLoaded;
        }

        std::unique_ptr<CUDADriverLibrary> cudaApi  = std::make_unique<CUDADriverLibrary>();
        std::unique_ptr<NVRTCLibrary>      nvrtcApi = std::make_unique<NVRTCLibrary>();
        dmt::os::LibraryLoader             loader{true};

        CUcontext cuCtx = 0;

    private:
        bool m_cudaLoaded  = false;
        bool m_nvrtcLoaded = false;
    } j;


    {
        dmt::Context ctx;
        ctx.trace("Hello Cruel World", {});

        dmt::os::Path   path = dmt::os::Path::fromString(DMT_PROJ_PATH "/scenes/scene_example.json");
        dmt::Parameters params{};
        dmt::Scene      scene;
        {
            dmt::Renderer renderer;
            dmt::Parser   parser{path, &renderer};
            if (!parser.parse())
            {
                ctx.error("Error parsing", {});
                return -1;
            }

            params = std::move(renderer.params);
            scene  = std::move(renderer.scene);
        }

        if (!j.loadCUDA())
        {
            ctx.error("Couldn't load CUDA Driver API", {});
            return 1;
        }

        // create cuda driver context
        CUdevice device = -1;
        j.cudaApi->cuInit(0);
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuDeviceGet(&device, 0)))
            return 1;
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxCreate(&j.cuCtx, 0, device)))
            return 1;

        if (!j.loadNVRTC())
        {
            ctx.error("Couldn't load NVRTC Library", {});
            return -1;
        }

        nvrtcProgram prog = 0;
        if (j.nvrtcApi->nvrtcCreateProgram(&prog, s_saxpySrc, "saxpy.cu", 0, nullptr, nullptr) != ::NVRTC_SUCCESS)
        {
            ctx.error("Couldn't create program", {});
            return -1;
        }

        if (j.nvrtcApi->nvrtcCompileProgram(prog, s_nvccOpts.size(), s_nvccOpts.data()) != ::NVRTC_SUCCESS)
        {
            size_t logSize;
            j.nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            j.nvrtcApi->nvrtcGetProgramLog(prog, log.data());
            ctx.error("NVRTC Compilation Failed: {}", std::make_tuple(log));
            j.nvrtcApi->nvrtcDestroyProgram(&prog);
            return 1;
        }

        size_t cubinSize = 0;
        j.nvrtcApi->nvrtcGetCUBINSize(prog, &cubinSize);
        std::unique_ptr<char[]> cubinBuffer = std::make_unique<char[]>(cubinSize);

        j.nvrtcApi->nvrtcGetCUBIN(prog, cubinBuffer.get());
        j.nvrtcApi->nvrtcDestroyProgram(&prog);

        CUmodule   module = nullptr;
        CUfunction func   = nullptr;
        j.cudaApi->cuModuleLoadData(&module, cubinBuffer.get());

        j.cudaApi->cuModuleGetFunction(&func, module, "saxpy_grid_stride");

        // Example: launch parameters
        int    n = 1024 * 1024;
        float  a = 2.0f;
        float* d_x; // device pointer to x
        float* d_y; // device pointer to y

        // Allocate device memory (example)
        j.cudaApi->cuMemAlloc((CUdeviceptr*)&d_x, n * sizeof(float));
        j.cudaApi->cuMemAlloc((CUdeviceptr*)&d_y, n * sizeof(float));

        // Kernel launch parameters
        uint32_t threadsPerBlock = 256;
        uint32_t blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Arguments array
        void* kernelArgs[] = {&n, &a, &d_x, &d_y};

        // Launch the kernel
        CUresult res = j.cudaApi->cuLaunchKernel(func,
                                                 blocksPerGrid,
                                                 1,
                                                 1, // grid dimensions
                                                 threadsPerBlock,
                                                 1,
                                                 1, // block dimensions
                                                 0, // shared memory
                                                 0, // stream
                                                 kernelArgs,
                                                 nullptr // extra (deprecated)
        );
        if (res != CUDA_SUCCESS)
        {
            std::cerr << "Failed to launch kernel" << std::endl;
            return 1;
        }

        // Wait for completion
        j.cudaApi->cuCtxSynchronize();

        // Clean up
        j.cudaApi->cuModuleUnload(module);
        j.cudaApi->cuMemFree((CUdeviceptr)d_x);
        j.cudaApi->cuMemFree((CUdeviceptr)d_y);
    }

    return 0;
}
