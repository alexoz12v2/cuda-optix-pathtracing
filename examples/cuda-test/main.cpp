#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"

#include "core/core-parser.h"
#include "core/core-render.h"
#include "core/core-bvh-builder.h"
#include "core/core-primitive.h"


#include "cuda-wrappers/cuda-wrappers-cuda-driver.h"
#include "cuda-wrappers/cuda-wrappers-utils.h"
#include "cuda-wrappers/cuda-wrappers-nvrtc.h"
#include "cudautils/cudautils-camera.h"
#include "cudautils/cudautils-bvh.h"
#include "cudautils/cudautils-kernels.h"
#include "cudautils/cudautils-pools.h"
#include "cudautils/cudautils-light.h"
#include "cudautils/cudautils-filter.h"

#include <algorithm>
#include <cstdint>

/*
 * Notes:
 *
 * Make work queues:
 * - PrimatyRayQueue
 * - RayIntersectQueue
 * - ShadeQueue
 * - ShadowRayQueue
 * - NextBOunceQueue
 * - FilmWriteQueue
 *
 * Single ray is map to one warp this for divergence and coalescing
 */

namespace /*static*/ {
    using namespace dmt;
    //---kernels declarations---
    static char const* kernerls = R"a(
extern "C" {
void kRayGen(DeviceCamera cam,int,int,int,int,int,RayPool,IndexQueue);
void kIntersect(DeviceBVH,DeviceLights,FilmSOA,RayPool,HitPool,IndexQueue,IndexQueue);
void kShade(DeviceBVH,DeviceLights,DeviceMaterials,FilmSOA,RayPool,HitPool,int,IndexQueue,IndexQueue);
}
)a";

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

//TODO: move to another file
namespace dmt {
    struct TriangleMeshSOA
    {
        float* px;
        float* py;
        float* pz;
        int*   indicesx;
        int*   indicesy;
        int*   indicesz;
        float* u;
        float* v;
        float* normalx;
        float* normaly;
        float* normalz;
    };


    template <typename T>
    void dalloc(CUDADriverLibrary* cudaApi, DeviceBuffer<T>& b, uint32_t n);
    void dalloc(CUDADriverLibrary* cudaApi, IndexQueue& q, uint32_t n);

    DeviceFilter buildAndUploadFilterHost(CUDADriverLibrary* cudaApi, filtering::Mitchell const& filter)
    {
        int const NumSamplesPerAxisPerDomainUnit = 32;

        // Use your exact class' radius() and evaluate()
        auto [rx, ry] = filter.radius();

        int Nx = std::max(1, static_cast<int>((NumSamplesPerAxisPerDomainUnit * rx)));
        int Ny = std::max(1, static_cast<int>((NumSamplesPerAxisPerDomainUnit * ry)));

        std::vector<float> h_f(Nx * Ny);

        float xMin = -rx, xMax = rx;
        float yMin = -ry, yMax = ry;

        // Tabulate filter exactly with your evaluate()
        for (int iy = 0; iy < Ny; ++iy)
        {
            float vy = (iy + 0.5f) / Ny;
            float y  = yMin + vy * (yMax - yMin);
            for (int ix = 0; ix < Nx; ++ix)
            {
                float vx          = (ix + 0.5f) / Nx;
                float x           = xMin + vx * (xMax - xMin);
                float v           = filter.evaluate(x, y); // your evaluate()
                h_f[iy * Nx + ix] = v > 0.f ? v : 0.f;
            }
        }

        // Build row CDFs and marginal CDFs
        std::vector<float> h_cond((Nx + 1) * Ny);
        std::vector<float> rowSums(Ny);

        for (int iy = 0; iy < Ny; ++iy)
        {
            h_cond[iy * (Nx + 1)] = 0.f;
            float s               = 0.f;
            for (int ix = 0; ix < Nx; ++ix)
            {
                s += h_f[iy * Nx + ix];
                h_cond[iy * (Nx + 1) + (ix + 1)] = s;
            }
            rowSums[iy] = s;
        }

        std::vector<float> h_marg(Ny + 1);
        h_marg[0]      = 0.f;
        float integral = 0.f;
        for (int iy = 0; iy < Ny; ++iy)
        {
            integral += rowSums[iy];
            h_marg[iy + 1] = integral;
        }

        // Normalize to [0,1]
        for (int iy = 0; iy < Ny; ++iy)
        {
            float rs = rowSums[iy];
            if (rs > 0.f)
            {
                for (int k = 0; k <= Nx; ++k)
                    h_cond[iy * (Nx + 1) + k] /= rs;
            }
            else
            {
                for (int k = 0; k <= Nx; ++k)
                    h_cond[iy * (Nx + 1) + k] = float(k) / float(Nx);
            }
        }
        for (int k = 0; k <= Ny; ++k)
            h_marg[k] /= integral;

        // Upload to device
        DeviceFilter df{};
        df.Nx       = Nx;
        df.Ny       = Ny;
        df.xMin     = xMin;
        df.xMax     = xMax;
        df.yMin     = yMin;
        df.yMax     = yMax;
        df.integral = integral;

        cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&df.d_f), sizeof(float) * h_f.size());
        cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(df.d_f), h_f.data(), sizeof(float) * h_f.size());

        cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&df.d_condCdf), sizeof(float) * h_cond.size());
        cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(df.d_condCdf), h_cond.data(), sizeof(float) * h_cond.size());

        cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&df.d_marginalCdf), sizeof(float) * h_marg.size());
        cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(df.d_marginalCdf), h_marg.data(), sizeof(float) * h_marg.size());

        return df;
    }

    UniqueRef<BVHWiVeCluster[]> bvhBuild(Scene const& scene, uint32_t* nodeCount)
    {
        if (scene.instances.size() > 0)
        {
            std::pmr::vector<BVHBuildNode*>        perInstanceBvhNodes{};
            std::pmr::vector<UniqueRef<Primitive>> primitives{};
            std::pmr::memory_resource*             memHeap = std::pmr::get_default_resource();
            perInstanceBvhNodes.reserve(64);
            primitives.reserve(256);

            for (size_t instanceIdx = 0; instanceIdx < scene.instances.size(); ++instanceIdx)
            {
                perInstanceBvhNodes.push_back(bvh::buildForInstance(scene, instanceIdx, primitives, memHeap, memHeap));
            }

            BVHBuildNode* bvhRoot = nullptr;
            if (perInstanceBvhNodes.size() > 1)
            {
                bvhRoot = reinterpret_cast<BVHBuildNode*>(memHeap->allocate(sizeof(BVHBuildNode)));
                std::memset(bvhRoot, 0, sizeof(BVHBuildNode));
                bvh::buildCombined(bvhRoot, perInstanceBvhNodes, memHeap, memHeap);
            }
            else
            {
                bvhRoot = perInstanceBvhNodes[0];
            }

            std::pmr::monotonic_buffer_resource tmp{4096};

            BVHWiVeCluster* wivebvh = bvh::buildBVHWive(bvhRoot, nodeCount, &tmp, memHeap);

            if (perInstanceBvhNodes.size() > 1)
            {
                bvh::cleanup(bvhRoot, memHeap);
            }

            return UniqueRef<BVHWiVeCluster[]>{wivebvh, PmrDeleter::create<BVHWiVeCluster[]>(memHeap, *nodeCount)};
        }
        else
        {
            *nodeCount = 0;
            return nullptr;
        }
    }

    int launchRayGen(CUDADriverLibrary*  cudaApi,
                     NVRTCLibrary*       nvrtcApi,
                     DeviceCamera const& cam,
                     int                 sx,
                     int                 sy,
                     int                 tw,
                     int                 th,
                     int                 spp,
                     RayPool             rays,
                     IndexQueue          rayQ);
    int launchIntersect(
        std::unique_ptr<CUDADriverLibrary> cudaApi,
        std::unique_ptr<char[]>            cubinBuffer,
        BVHWiVeCluster                     bvh,
        int                                nodeCount,
        DeviceLights                       lights,
        FilmSOA                            film,
        RayPool                            rays,
        HitPool                            hits,
        GpuHaltonOwenSampler*              samplers,
        IndexQueue                         inRayQ,
        IndexQueue                         shadeQ);
    int launchShade(CUDADriverLibrary* cudaApi,
                    NVRTCLibrary*      nvrtcApi,
                    BVHWiVeCluster     bvh,
                    DeviceLights       lights,
                    DeviceMaterials    mats,
                    FilmSOA            film,
                    RayPool            rays,
                    HitPool            hits,
                    int                maxDepth,
                    IndexQueue         shadeQ,
                    IndexQueue         nextRayQ);

    //function to allocate in global memory
    template <typename T>
    void dalloc(CUDADriverLibrary* cudaApi, DeviceBuffer<T>& b, uint32_t n)
    {
        b.capacity = n;
        cudaMalloc(&b.ptr, sizeof(T) * n);
    }

    //the allocate from the global memory by the queue
    void dalloc(CUDADriverLibrary* cudaApi, IndexQueue& q, uint32_t n)
    {
        q.capacity = n;
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&q.items), sizeof(uint32_t) * n);
        cudaApi->cuMemAlloc(reinterpret_cast<unsigned long long*>(&q.tail), sizeof(uint32_t));
        cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(q.tail), 0, n);
    }

    //allloacte functions for the several pools

    //---lunch wrappers---
    //
    //handle a ray for each wrappers
    int launchRayGen(CUDADriverLibrary* cudaApi,
                     NVRTCLibrary*      nvrtcApi,
                     DeviceCamera&      cam,
                     int                sx,
                     int                sy,
                     int                tw,
                     int                th,
                     int                spp,
                     RayPool            rays,
                     IndexQueue         rayQ)
    {
        nvrtcProgram prog = 0;
        if (nvrtcApi->nvrtcCreateProgram(&prog, s_saxpySrc, "saxpy.cu", 0, nullptr, nullptr) != ::NVRTC_SUCCESS)
        {
            return -1;
        }

        if (nvrtcApi->nvrtcCompileProgram(prog, s_nvccOpts.size(), s_nvccOpts.data()) != ::NVRTC_SUCCESS)
        {
            size_t logSize;
            nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            nvrtcApi->nvrtcGetProgramLog(prog, log.data());
            nvrtcApi->nvrtcDestroyProgram(&prog);
            return 1;
        }

        size_t cubinSize = 0;
        nvrtcApi->nvrtcGetCUBINSize(prog, &cubinSize);
        std::unique_ptr<char[]> cubinBuffer = std::make_unique<char[]>(cubinSize);
        // simple: 128 threads per block
        int total = tw * th * spp;
        int block = 128;
        int grid  = (total + block - 1) / block;
        //REDO
        //kRayGen<<<grid, block>>>(cam, sx, sy, tw, th, spp, rays, rayQ);
        CUmodule   module = nullptr;
        CUfunction func   = nullptr;
        cudaApi->cuModuleLoadData(&module, cubinBuffer.get());

        void* kernelArgs[] = {&cam, &sx, &sy, &tw, &th, &spp, &rays, &rayQ};

        cudaApi->cuModuleGetFunction(&func, module, "saxpy_grid_stride");
        CUresult res = cudaApi->cuLaunchKernel(func,
                                               grid,
                                               1,
                                               1, // grid dimensions
                                               block,
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
    }

    int launchIntersect(CUDADriverLibrary* cudaApi,
                        NVRTCLibrary*      nvrtcApi,
                        BVHWiVeCluster     bvh,
                        DeviceLights       lights,
                        FilmSOA            film,
                        RayPool            rays,
                        HitPool            hits,
                        IndexQueue         inRayQ,
                        IndexQueue         shadeQ)
    {
        nvrtcProgram prog = 0;
        if (nvrtcApi->nvrtcCreateProgram(&prog, s_saxpySrc, "saxpy.cu", 0, nullptr, nullptr) != ::NVRTC_SUCCESS)
        {
            return -1;
        }

        if (nvrtcApi->nvrtcCompileProgram(prog, s_nvccOpts.size(), s_nvccOpts.data()) != ::NVRTC_SUCCESS)
        {
            size_t logSize;
            nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            nvrtcApi->nvrtcGetProgramLog(prog, log.data());
            nvrtcApi->nvrtcDestroyProgram(&prog);
            return 1;
        }

        size_t cubinSize = 0;
        nvrtcApi->nvrtcGetCUBINSize(prog, &cubinSize);
        std::unique_ptr<char[]> cubinBuffer = std::make_unique<char[]>(cubinSize);

        nvrtcApi->nvrtcGetCUBIN(prog, cubinBuffer.get());
        nvrtcApi->nvrtcDestroyProgram(&prog);
        // read size
        uint32_t qsize = 0;
        cudaApi->cuMemcpyDtoH(&qsize, reinterpret_cast<unsigned long long>(&inRayQ.tail), sizeof(uint32_t));
        if (qsize == 0)
            return 1;
        int block = 128;
        int grid  = (qsize + block - 1) / block;
        //kIntersect<<<grid, block>>>(bvh, lights, film, rays, hits, inRayQ, shadeQ);
        CUmodule   module = nullptr;
        CUfunction func   = nullptr;
        cudaApi->cuModuleLoadData(&module, cubinBuffer.get());

        void* kernelArgs[] = {&bvh, &lights, &film, &rays, &hits, &inRayQ, &shadeQ};

        cudaApi->cuModuleGetFunction(&func, module, "saxpy_grid_stride");
        CUresult res = cudaApi->cuLaunchKernel(func,
                                               grid,
                                               1,
                                               1, // grid dimensions
                                               block,
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
    }

    int launchShade(CUDADriverLibrary* cudaApi,
                    NVRTCLibrary*      nvrtcApi,
                    BVHWiVeCluster     bvh,
                    DeviceLights       lights,
                    DeviceMaterials    mats,
                    FilmSOA            film,
                    RayPool            rays,
                    HitPool            hits,
                    int                maxDepth,
                    IndexQueue         shadeQ,
                    IndexQueue         nextRayQ)
    {
        nvrtcProgram prog = 0;
        if (nvrtcApi->nvrtcCreateProgram(&prog, s_saxpySrc, "saxpy.cu", 0, nullptr, nullptr) != ::NVRTC_SUCCESS)
        {
            return -1;
        }

        if (nvrtcApi->nvrtcCompileProgram(prog, s_nvccOpts.size(), s_nvccOpts.data()) != ::NVRTC_SUCCESS)
        {
            size_t logSize;
            nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            nvrtcApi->nvrtcGetProgramLog(prog, log.data());
            nvrtcApi->nvrtcDestroyProgram(&prog);
            return 1;
        }

        size_t cubinSize = 0;
        nvrtcApi->nvrtcGetCUBINSize(prog, &cubinSize);
        std::unique_ptr<char[]> cubinBuffer = std::make_unique<char[]>(cubinSize);

        nvrtcApi->nvrtcGetCUBIN(prog, cubinBuffer.get());
        nvrtcApi->nvrtcDestroyProgram(&prog);
        uint32_t qsize = 0;
        cudaApi->cuMemcpyDtoH(&qsize, reinterpret_cast<unsigned long long>(&shadeQ.tail), sizeof(uint32_t));
        if (qsize == 0)
            return 1;
        int block = 128;
        int grid  = (qsize + block - 1) / block;
        //kShade<<<grid, block>>>(bvh, lights, mats, film, rays, hits, maxDepth, shadeQ, nextRayQ);
        CUmodule   module = nullptr;
        CUfunction func   = nullptr;
        cudaApi->cuModuleLoadData(&module, cubinBuffer.get());

        void* kernelArgs[] = {&bvh, &lights, &mats, &film, &rays, &hits, &maxDepth, &shadeQ, &nextRayQ};

        cudaApi->cuModuleGetFunction(&func, module, "saxpy_grid_stride");
        CUresult res = cudaApi->cuLaunchKernel(func,
                                               grid,
                                               1,
                                               1, // grid dimensions
                                               block,
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

        int                       SPP = 4, MaxDepth = 5;
        static constexpr uint32_t ScratchBufferBytes = 4096;

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
        //-------------camera------------------
        //host
        DeviceCamera cam{};
        cam.focalLength = params.focalLength;
        cam.sensorSize  = params.sensorSize;
        //cam.camDirX     = params.cameraDirection.x;
        //cam.camDirY     = params.cameraDirection.y;
        //cam.camDirZ     = params.cameraDirection.z;
        //cam.camPosX     = params.cameraPosition.x;
        //cam.camPosY     = params.cameraPosition.y;
        //cam.camPosZ     = params.cameraPosition.z;
        cam.dir.x = params.cameraDirection.x;
        cam.dir.y = params.cameraDirection.y;
        cam.dir.z = params.cameraDirection.z;
        cam.pos.x = params.cameraPosition.x;
        cam.pos.y = params.cameraPosition.y;
        cam.pos.z = params.cameraPosition.z;
        cam.spp   = SPP;

        //device
        DeviceCamera* dcam;
        j.cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&cam), sizeof(DeviceCamera));
        j.cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dcam), &cam, sizeof(DeviceCamera));
        //---------BVH--------------------------
        uint32_t                         bvhNodeCount = 0;
        dmt::UniqueRef<BVHWiVeCluster[]> bvh          = bvhBuild(scene, &bvhNodeCount);
        //uploading to the GPU the BVH
        BVHWiVeCluster* dbvh = nullptr;
        j.cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&dbvh), bvhNodeCount * sizeof(BVHWiVeCluster));
        j.cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dbvh), bvh.get(), bvhNodeCount * sizeof(BVHWiVeCluster));
        DeviceLights    dlights{}; // upload) light tree/env
        DeviceMaterials dmats{};   // upload materials/textures

        // 2) Allocate film, pools, and queues
        //---film-----
        FilmSOA film;
        dmt::allocFilm(j.cudaApi.get(), film, params.filmResolution.x, params.filmResolution.y);
        //---rays-----
        uint32_t numRays = film.width * film.height * SPP;
        RayPool  rays;
        dmt::allocRayPool(j.cudaApi.get(), rays, numRays); // capacity = total primary rays
        //---Hits-----
        HitPool hits;
        dmt::allocHitPool(j.cudaApi.get(), hits, rays.capacity);
        //---queue----
        IndexQueue rayQ{}, shadeQ{}, nextRayQ{};
        dmt::dalloc(j.cudaApi.get(), rayQ, rays.capacity);
        dmt::dalloc(j.cudaApi.get(), shadeQ, rays.capacity);
        dmt::dalloc(j.cudaApi.get(), nextRayQ, rays.capacity);
        //----filter---
        auto scratchBuffer = makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), ScratchBufferBytes);
        std::pmr::monotonic_buffer_resource    scratch{scratchBuffer.get(), ScratchBufferBytes};
        std::pmr::unsynchronized_pool_resource pool;
        filtering::Mitchell                    filter{{{2.f, 2.f}}, 1.f / 3.f, 1.f / 3.f, &pool, &scratch};
        DeviceFilter                           dfilter = buildAndUploadFilterHost(j.cudaApi.get(), filter);
        //-------sampler-----------
        GpuHaltonOwenSampler* dsamplers;
        j.cudaApi->cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&dsamplers),
                              film.width * film.height * SPP * sizeof(GpuHaltonOwenSampler));
        //Initialize each sampler with starting sample index
        std::pmr::monotonic_buffer_resource    buffer;
        std::pmr::vector<GpuHaltonOwenSampler> hsamplers{&buffer};

        for (int i = 0; i < numRays; i++)
        {
            GpuHaltonOwenSampler s;
            s.startSample(0);
            hsamplers.push_back(s);
        }

        //copy to GPU
        j.cudaApi->cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dsamplers),
                                hsamplers.data(),
                                sizeof(GpuHaltonOwenSampler) * hsamplers.size());
        // 3) Tiling
        int const TileW = 32, TileH = 32;
        int       numTileX = (film.width + TileW - 1) / TileW;
        int       numTileY = (film.height + TileH - 1) / TileH;

        for (int ty = 0; ty < numTileY; ++ty)
        {
            for (int tx = 0; tx < numTileX; ++tx)
            {
                int startX = tx * TileW;
                int startY = ty * TileH;
                int tw     = std::min(TileW, film.width - startX);
                int th     = std::min(TileH, film.height - startY);

                // reset queues for this tile
                j.cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(rayQ.tail), 0, sizeof(uint32_t));
                j.cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(shadeQ.tail), 0, sizeof(uint32_t));
                j.cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(nextRayQ.tail), 0, sizeof(uint32_t));

                // 4) RayGen
                dmt::launchRayGen(j.cudaApi.get(), j.nvrtcApi.get(), dcam, startX, startY, tw, th, SPP, rays, rayQ);
                //wait the end of the raygen kernel
                j.cudaApi->cuCtxSynchronize();

                // 5) Bounce loop (wavefront): Intersect ? Shade; swap queues
                for (int depth = 0; depth < MaxDepth; ++depth)
                {
                    // Intersect current rayQ ? shadeQ
                    j.cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(shadeQ.tail), 0, sizeof(uint32_t));
                    dmt::launchIntersect(j.cudaApi.get(), j.nvrtcApi.get(), dbvh, dlights, film, rays, hits, rayQ, shadeQ);
                    j.cudaApi->cuCtxSynchronize();

                    // Shade shadeQ ? nextRayQ
                    j.cudaApi->cuMemsetD32(reinterpret_cast<unsigned long long>(nextRayQ.tail), 0, sizeof(uint32_t));
                    dmt::launchShade(j.cudaApi.get(), j.nvrtcApi.get(), dbvh, dlights, dmats, film, rays, hits, MaxDepth, shadeQ, nextRayQ);
                    j.cudaApi->cuCtxSynchronize();

                    // Check if any next rays exist
                    uint32_t nNext = 0;
                    j.cudaApi->cuMemcpyDtoH(&nNext, reinterpret_cast<unsigned long long>(nextRayQ.tail), sizeof(uint32_t));
                    if (nNext == 0)
                        break;

                    // Swap rayQ <- nextRayQ
                    std::swap(rayQ.items, nextRayQ.items);
                    std::swap(rayQ.tail, nextRayQ.tail);
                    std::swap(rayQ.capacity, nextRayQ.capacity);
                }
            }
        }


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
