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

        // OptixProgramJanitor scope
        {
            struct OptixProgramJanitor
            {
                explicit OptixProgramJanitor(Janitor& _j) : j(_j) {}
                ~OptixProgramJanitor() noexcept
                {
                    if (d_gasBuffer)
                        j.cudaApi->cuMemFree(d_gasBuffer);
                }

                Janitor&    j;
                CUdeviceptr d_gasBuffer = 0;
            } oj{j};

            // -- 1. Acceleration Structure Definition --
            auto        primitives     = dmt::makeUniqueRef<OptixBuildInput[]>(std::pmr::get_default_resource(), 1);
            CUdeviceptr d_sphereBuffer = 0;

            // a sphere is defined in device memory by a center (float3) and radius (float)
            float const sphereData[4]{0.f, 0.f, 0.f, 1.5f};
            if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&d_sphereBuffer, 4 * sizeof(float))))
                return 1;

            if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemcpyHtoD(d_sphereBuffer, sphereData, 4 * sizeof(float))))
                return 1;
            CUdeviceptr d_radiusBuffer = d_sphereBuffer + 3 * sizeof(float);

            primitives[0]                           = {};
            primitives[0].type                      = ::OPTIX_BUILD_INPUT_TYPE_SPHERES;
            primitives[0].sphereArray.vertexBuffers = &d_sphereBuffer;
            primitives[0].sphereArray.numVertices   = 1;
            primitives[0].sphereArray.radiusBuffers = &d_radiusBuffer;

            // Prepare 1 record for the Shader Binding Table
            uint32_t sphereInputFlags[1]            = {::OPTIX_GEOMETRY_FLAG_NONE};
            primitives[0].sphereArray.flags         = sphereInputFlags;
            primitives[0].sphereArray.numSbtRecords = 1;

            // build options:
            OptixAccelBuildOptions accelOptions{};
            accelOptions.buildFlags = ::OPTIX_BUILD_FLAG_ALLOW_COMPACTION | ::OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accelOptions.operation = ::OPTIX_BUILD_OPERATION_BUILD;

            // estimate 3 memory usage metrics for the acceleration struture
            // (size in bytes, temp size in bytes (OPTIX_BUILD_OPERATION_BUILD, non compacted), temp update size in bytes (UPDATE, non compacted))
            OptixAccelBufferSizes gasBufferSizes{};
            if (!dmt::optixCall(
                    optixAccelComputeMemoryUsage(j.optixContext, &accelOptions, primitives.get(), 1, &gasBufferSizes)))
                return 1;

            // allocate the temp buffer used by OptiX in the build operation
            CUdeviceptr d_tempBuffer = 0;
            if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&d_tempBuffer, gasBufferSizes.tempSizeInBytes)))
                return 1;

            // allocate sufficient device memory to host the non-compacted geometry acceleration structure
            // (last 8 bytes are written as the compacted size in the build operation, overallocation due to alignment requirements)
            CUdeviceptr d_tempOutputGasAndCompactedSize = 0;
            if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                     j.cudaApi->cuMemAlloc(&d_tempOutputGasAndCompactedSize,
                                                           gasBufferSizes.outputSizeInBytes + 2 * sizeof(size_t))))
                return 1;

            // define where should the compacted size be emitted during the build (ie last 8 bytes of the aformentioned buffer)
            // emitted properties must be 8 bytes aligned in device memory
            constexpr auto alignUp8 = [](uintptr_t value) -> uintptr_t {
                return (value + 7) & ~static_cast<uintptr_t>(7);
            };
            OptixAccelEmitDesc emitProperty{};
            emitProperty.type   = ::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = static_cast<CUdeviceptr>(
                alignUp8(static_cast<uintptr_t>(d_tempOutputGasAndCompactedSize + gasBufferSizes.outputSizeInBytes)));

            OptixTraversableHandle gasHandle = 0;

            // perform the build operation
            // clang-format off
            if (!dmt::optixCall(optixAccelBuild(
                j.optixContext, 0/*stream*/, &accelOptions, 
                primitives.get(), 1,                             // build inputs
                d_tempBuffer, gasBufferSizes.tempSizeInBytes, // temp buffer
                d_tempOutputGasAndCompactedSize, gasBufferSizes.outputSizeInBytes, // output buffer
                &gasHandle,       // output handle
                &emitProperty, 1 // emitted properties
            )))
                return 1;
            // clang-format on

            // clean up everything except for the uncompacted GAS buffer
            if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemFree(d_sphereBuffer)) ||
                !dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemFree(d_tempBuffer)))
                return 1;

            // eventually update the Acceleration structure with more elements ...
            // ... then, retrieve compacted size, allocate buffer and compact the AS
            size_t gasCompactedSize = 0;
            if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                     j.cudaApi->cuMemcpyDtoH(&gasCompactedSize, emitProperty.result, sizeof(size_t))))
                return 1;

            if (gasCompactedSize < gasBufferSizes.outputSizeInBytes)
            {
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&oj.d_gasBuffer, gasCompactedSize)))
                    return 1;

                if (!dmt::optixCall(
                        optixAccelCompact(j.optixContext, 0 /*stream*/, gasHandle, oj.d_gasBuffer, gasCompactedSize, &gasHandle)))
                    return 1;

                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemFree(d_tempOutputGasAndCompactedSize)))
                    return 1;
            }
            else
                oj.d_gasBuffer = d_tempOutputGasAndCompactedSize;

            ctx.log("Built and Compacted Acceleration Structure", {});
            // Memory Leak!

            // -- 2: Module Creation --
        }
    }

    return 0;
}