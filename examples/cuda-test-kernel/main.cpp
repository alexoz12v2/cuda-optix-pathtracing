#include "cuda-test.h"
#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"

#include "cuda-wrappers/cuda-wrappers-cuda-driver.h"
#include "cuda-wrappers/cuda-wrappers-utils.h"
#include "cuda-wrappers/cuda-wrappers-nvrtc.h"

#include "cuda-queue.h"
#include "cuda-test.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <fstream>
#include <iterator>

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor()
        {
            if (cuCtx && !cuCtxIsPrimary)
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

        CUcontext cuCtx          = 0;
        bool      cuCtxIsPrimary = false;

    private:
        bool m_cudaLoaded  = false;
        bool m_nvrtcLoaded = false;
    } j;


    {
        dmt::Context ctx;
        ctx.trace("Hello Cruel World", {});

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
        //if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxCreate(&j.cuCtx, 0, device)))
        //    return 1;
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuDevicePrimaryCtxRetain(&j.cuCtx, device)))
            return 1;
        j.cuCtxIsPrimary = true;

        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxSetCurrent(j.cuCtx)))
            return 1;

        unsigned int flags;
        int          active;
        j.cudaApi->cuDevicePrimaryCtxGetState(device, &flags, &active);
        ctx.log("active = {} (should be one), flags = {}\n", std::make_tuple(active, flags));
        unsigned int vers;
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxGetApiVersion(j.cuCtx, &vers));

        int major = 0;
        int minor = 0;
        j.cudaApi->cuDeviceComputeCapability(&major, &minor, device);
        ctx.log("CUDA Compute Capability: {}.{}", std::make_tuple(major, minor));

        if (!j.loadNVRTC())
        {
            ctx.error("Couldn't load NVRTC Library", {});
            return -1;
        }

        if (ctx.isTraceEnabled())
        {
            ctx.trace("NVRTC Supported arch:", {});
            int32_t supported = 0;
            j.nvrtcApi->nvrtcGetNumSupportedArchs(&supported);
            auto psupp = std::make_unique<int[]>(supported);
            j.nvrtcApi->nvrtcGetSupportedArchs(psupp.get());

            for (int32_t i = 0; i < supported; ++i)
            {
                ctx.trace("  compute_{}", std::make_tuple(psupp[i]));
            }
        }

        dmt::os::Path path       = dmt::os::Path::executableDir() / "shaders" / "cuda-kernel.cu";
        auto          nvccOpts   = dmt::getnvccOpts(true);
        std::string   includeOpt = "--include-path=";
        includeOpt.append((dmt::os::Path::executableDir() / "shaders").toUnderlying());
        nvccOpts.push_back(includeOpt.c_str());

        std::unique_ptr<char[]> saxpyPTX = dmt::compilePTX(path, j.nvrtcApi.get(), "saxpy.cu", nvccOpts);

        CUmodule   mod  = nullptr;
        CUfunction func = nullptr;

        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuModuleLoadData(&mod, saxpyPTX.get()));
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuModuleGetFunction(&func, mod, "saxpy_grid_stride"));

        // Example: launch parameters
        int    n = 1024;
        float  a = 2.0f;
        float* d_x; // device pointer to x
        float* d_y; // device pointer to y

        // Allocate device memory (example)
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc((CUdeviceptr*)&d_x, n * sizeof(float)));
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc((CUdeviceptr*)&d_y, n * sizeof(float)));

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
            char const* err = nullptr;
            j.cudaApi->cuGetErrorString(res, &err);
            ctx.error("Error cuLaunchKernel saxpy: {}", std::make_tuple(std::string_view(err)));
        }

// Wait for completion
#if defined(DMT_OS_LINUX)
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxSynchronize());
#elif defined(DMT_OS_WINDOWS)
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxSynchronize(j.cuCtx));
#else
    #error "unsupported OS"
#endif

        // Clean up
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemFree((CUdeviceptr)d_x));
        dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemFree((CUdeviceptr)d_y));

        // testing queues
        ctx.log("--- Testing Queues ---", {});
        size_t queueBytes  = 0;
        size_t queueBytes1 = 0;
        auto*  queue       = dmt::ManagedQueue<int>::allocateManaged(*j.cudaApi, 256, queueBytes);
        auto*  queue1      = dmt::ManagedQueue<int>::allocateManaged(*j.cudaApi, 256, queueBytes1);
        if (!queue || !queue1)
        {
            ctx.error("Error allocating queue", {});
        }
        else
        {
            ctx.log("Allocated Queue in __managed__ memory with attach to host strategy", {});
            for (int i = 0; i < queue->capacity; ++i)
                queue->pushHost(i);

            CUfunction kqueueDouble = nullptr;
            dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuModuleGetFunction(&kqueueDouble, mod, "kqueueDouble"));
            CUstream stream;
            dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuStreamCreate(&stream, CU_STREAM_DEFAULT));
            // attach __managed__ memory to stream

            dmt::cudaDriverCall(j.cudaApi.get(),
                                j.cudaApi->cuStreamAttachMemAsync(stream,
                                                                  std::bit_cast<CUdeviceptr>(queue),
                                                                  queueBytes,
                                                                  CU_MEM_ATTACH_SINGLE));
            dmt::cudaDriverCall(j.cudaApi.get(),
                                j.cudaApi->cuStreamAttachMemAsync(stream,
                                                                  std::bit_cast<CUdeviceptr>(queue1),
                                                                  queueBytes1,
                                                                  CU_MEM_ATTACH_SINGLE));

            CUmemLocation memLoc;
            memLoc.type = CU_MEM_LOCATION_TYPE_DEVICE;
            memLoc.id   = 1;

            // optional: prefetch
            // TODO:: check the getAttribute to verify the compatibilty with the concurrent managed access
            // int concurrentManagedAccess = 0;
            //cuDeviceGetAttribute(&concurrentManagedAccess,
            //             CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
            //             dev);
            //dmt::cudaDriverCall(j.cudaApi.get(),j.cudaApi->cuMemPrefetchAsync(std::bit_cast<CUdeviceptr>(queue), queueBytes >> 3, memLoc, 0, stream));
            //dmt::cudaDriverCall(j.cudaApi.get(),j.cudaApi->cuMemPrefetchAsync(std::bit_cast<CUdeviceptr>(queue), queueBytes1 >> 3, memLoc, 0, stream));

            // launch
            void* kArgs[] = {&queue, &queue1};
            dmt::cudaDriverCall(j.cudaApi.get(),
                                j.cudaApi->cuLaunchKernel(kqueueDouble, 1, 1, 1, 256, 1, 1, 0, stream, kArgs, nullptr));
            dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuStreamSynchronize(stream));

            int         element = 0;
            std::string log;
            while (queue1->popHost(&element))
            {
                log += std::to_string(element);
                log += ", ";
            }


            if (log.size() >= 2)
            {
                log.pop_back();
                log.pop_back();
            }

            ctx.log("Queue Elements: [ ", {});
            // Break the log into chunks of 256 characters
            size_t const chunkSize = 256;
            for (size_t i = 0; i < log.size(); i += chunkSize)
            {
                std::string_view chunk(log.data() + i, std::min(chunkSize, log.size() - i));
                ctx.log("{}", std::make_tuple(chunk));
            }
            ctx.log(" ]", {});
            dmt::ManagedQueue<int>::freeManaged(*j.cudaApi, queue);
            dmt::ManagedQueue<int>::freeManaged(*j.cudaApi, queue1);
        }

        // clean up
        j.cudaApi->cuModuleUnload(mod);
    }

    return 0;
}
