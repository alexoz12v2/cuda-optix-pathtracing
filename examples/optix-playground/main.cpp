#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "cuda-wrappers/cuda-nvrtc.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "optixSphere.h"
#include <optix_stack_size.h>
// maybe to remove
#include <optix_stubs.h>

#include <fstream>
#include <numbers>

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
            {
                ctx.error("OptiX Failed: [{}] {}",
                          std::make_tuple(std::string_view(optixGetErrorName(result)),
                                          std::string_view(optixGetErrorString(result))));
            }
            return false;
        }

        return true;
    }

    std::pair<UniqueRef<char*[]>, UniqueRef<char[]>> copyStringsToBuffer(
        std::pmr::vector<std::pmr::string> const& strings,
        size_t*                                   outNumBytes,
        std::pmr::memory_resource*                resource = std::pmr::get_default_resource())
    {
        if (!outNumBytes)
            return {nullptr, nullptr};

        size_t totalStringBytes = 0;
        for (auto const& str : strings)
            totalStringBytes += str.size() + 1; // include null terminator

        *outNumBytes = totalStringBytes;

        // Allocate buffer for raw characters
        UniqueRef<char[]> buffer = makeUniqueRef<char[]>(resource, totalStringBytes);
        if (!buffer)
            return {nullptr, nullptr};

        // Allocate array of char* pointers
        UniqueRef<char*[]> cStringArray = makeUniqueRef<char*[]>(resource, strings.size());
        if (!cStringArray)
            return {nullptr, nullptr};

        // Fill buffer and pointer array
        size_t offset = 0;
        for (size_t i = 0; i < strings.size(); ++i)
        {
            auto const& str = strings[i];
            size_t      len = str.size();
            std::memcpy(buffer.get() + offset, str.c_str(), len);
            buffer.get()[offset + len] = '\0'; // null terminate
            cStringArray.get()[i]      = buffer.get() + offset;
            offset += len + 1;
        }

        return {std::move(cStringArray), std::move(buffer)};
    }

} // namespace dmt

struct Camera
{
    Camera() : eye{1.0f, 1.0f, 1.0f}, lookAt{0.f, 0.f, 0.f}, up{0.f, 1.f, 0.f}, fovY(35.f), aspectRatio(1.f) {}
    void uvwFrame(float U[3], float V[3], float W[3]) const
    {
        constexpr auto length = [](float const vec[3]) -> float {
            return sqrtf(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        };

        constexpr auto normalize = [](float vec[3]) -> void {
            float len = sqrtf(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
            if (len > 0.f)
            {
                vec[0] /= len;
                vec[1] /= len;
                vec[2] /= len;
            }
        };

        constexpr auto cross = [](float const in0[3], float const in1[3], float out[3]) -> void {
            out[0] = in0[1] * in1[2] - in0[2] * in1[1];
            out[1] = in0[2] * in1[0] - in0[0] * in1[2];
            out[2] = in0[0] * in1[1] - in0[1] * in1[0];
        };

        constexpr auto scalarMul = [](float inOut[3], float s) -> void {
            inOut[0] *= s;
            inOut[1] *= s;
            inOut[2] *= s;
        };

        // Do not normalize W -- it implies focal length
        W[0] = lookAt[0] - eye[0];
        W[1] = lookAt[1] - eye[1];
        W[2] = lookAt[2] - eye[2];

        float wlen = length(W);
        cross(W, up, U);
        normalize(U);
        cross(U, W, V);
        normalize(V);

        float vlen = wlen * tanf(0.5f * fovY * std::numbers::pi_v<float> / 180.0f);
        scalarMul(V, vlen);
        float ulen = vlen * aspectRatio;
        scalarMul(U, ulen);
    }

    float eye[3];
    float lookAt[3];
    float up[3];
    float fovY;
    float aspectRatio;
};

static void configureCamera(Camera& inOutCamera, float width, float height)
{
    inOutCamera.eye[0]      = 0.f;
    inOutCamera.eye[1]      = 0.f;
    inOutCamera.eye[2]      = 3.f;
    inOutCamera.lookAt[0]   = 0.f;
    inOutCamera.lookAt[1]   = 0.f;
    inOutCamera.lookAt[2]   = 0.f;
    inOutCamera.up[0]       = 0.f;
    inOutCamera.up[1]       = 1.f;
    inOutCamera.up[2]       = 3.f;
    inOutCamera.fovY        = 60.f;
    inOutCamera.aspectRatio = width / height;
}

template <typename T>
struct SbtRecord
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord   = SbtRecord<RayGenData>;
using MissSbtRecord     = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

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

        float width = 600.f, height = 480.f;

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
                    std::string_view sTag(tag, strlen(tag));
                    std::string_view sMessage(message, strlen(message));
                    if (sTag.length() > 0 && sMessage.length())
                    {
                        if (level <= 2)
                            ctx.error("{}: {}\n\0", std::make_tuple(sTag, sMessage));
                        else if (level == 3)
                            ctx.warn("{}: {}\n\0", std::make_tuple(sTag, sMessage));
                        else
                            ctx.log("{}: {}\n\0", std::make_tuple(sTag, sMessage));
                    }
                }
            },
            .logCallbackData = nullptr,
            .logCallbackLevel = 4,
            .validationMode = ::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
        };
        // clang-format on

        // Disable caching just for the moment
        dmt::os::env::set("OPTIX_CACHE_MAXSIZE", "0");
        dmt::os::env::set("OPTIX_FORCE_DEPRECATED_LAUNCHER", "1"); // printf in OptiX kernels
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

            auto   logString     = dmt::makeUniqueRef<char[]>(std::pmr::get_default_resource(), 2048);
            size_t logStringSize = 2048;
            size_t logStringLen  = 2048;

            // -- 1. Acceleration Structure Definition --
            OptixTraversableHandle gasHandle = 0;
            {
                auto        primitives     = dmt::makeUniqueRef<OptixBuildInput[]>(std::pmr::get_default_resource(), 1);
                CUdeviceptr d_sphereBuffer = 0;

                // a sphere is defined in device memory by a center (float3) and radius (float)
                float const sphereData[4]{0.f, 0.f, 0.f, 1.5f};
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&d_sphereBuffer, 4 * sizeof(float))))
                    return 1;

                if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                         j.cudaApi->cuMemcpyHtoD(d_sphereBuffer, sphereData, 4 * sizeof(float))))
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
                if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                         j.cudaApi->cuMemAlloc(&d_tempBuffer, gasBufferSizes.tempSizeInBytes)))
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
            }

            // -- 2: Module Creation: Get Module, Intersection Module, Pipeline Compile Options --
            OptixModule                 sphereModule             = 0;
            OptixModule                 sphereIntersectionModule = 0;
            OptixPipelineCompileOptions pipelineCompileOptions{};
            {
                dmt::os::Path optixSpherePath        = dmt::os::Path::executableDir() / "shaders" / "optixSphere.cu";
                dmt::os::Path optixSphereHeaderDir   = dmt::os::Path::executableDir() / "shaders";
                dmt::os::Path optixHeaderDir         = dmt::os::Path::executableDir().parent() / "shaders";
                dmt::os::Path optixHeaderInternalDir = optixHeaderDir / "internal";

                std::pmr::string optixSphereSource = dmt::os::readFileContents(optixSpherePath);
                ctx.log("Read CUDA Sources into memory", {});

                nvrtcProgram prog     = 0;
                nvrtcResult  nvrtcRes = j.nvrtcApi->nvrtcCreateProgram(&prog,
                                                                      optixSphereSource.c_str(),
                                                                      "optixSphere.cu",
                                                                      0,
                                                                      nullptr,
                                                                      nullptr);
                if (nvrtcRes != ::NVRTC_SUCCESS)
                    return 1;

                // COMPILE
                std::pmr::vector<std::pmr::string> options{
                    "-std=c++20",
                    "-arch",
                    "compute_60",
                    "-optix-ir",
                    "-lineinfo",
                    "-G",
                    "--use_fast_math",
                    "-default-device",
                    "-rdc",
                    "true",
                    "--include-path=" +
#if defined(_WIN32)
                        optixHeaderDir.toUnderlying().substr(4),
#else
                        optixHeaderDir.toUnderlying(),
#endif
                    "--include-path=" +
#if defined(_WIN32)
                        optixSphereHeaderDir.toUnderlying().substr(4),
#else
                        optixSphereHeaderDir.toUnderlying(),
#endif
                };

                size_t optionsNumBytes           = 0;
                auto [stringArray, stringBuffer] = dmt::copyStringsToBuffer(options, &optionsNumBytes);

                nvrtcRes = j.nvrtcApi->nvrtcCompileProgram(prog, options.size(), stringArray.get());
                if (nvrtcRes != ::NVRTC_SUCCESS)
                {
                    size_t logSize;
                    j.nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
                    std::string log(logSize, '\0');
                    j.nvrtcApi->nvrtcGetProgramLog(prog, log.data());
                    ctx.error("NVRTC Compilation Failed: {}", std::make_tuple(log));
                    j.nvrtcApi->nvrtcDestroyProgram(&prog);
                    return 1;
                }

                size_t optixIRSize = 0;
                nvrtcRes           = j.nvrtcApi->nvrtcGetOptiXIRSize(prog, &optixIRSize);
                if (nvrtcRes != ::NVRTC_SUCCESS)
                {
                    auto cstrError = j.nvrtcApi->nvrtcGetErrorString(nvrtcRes);
                    ctx.error("NVRTC Error: {}", std::make_tuple(cstrError));
                    return 1;
                }
                else if (!optixIRSize)
                {
                    ctx.error(
                        "The value of nvrtcGetOptiXIRSize is set to 0 if the program was compiled with options "
                        "incompatible with OptiX IR generation.",
                        {});
                    return 1;
                }

                auto optixSphereOptixIR = dmt::makeUniqueRef<char[]>(std::pmr::get_default_resource(), optixIRSize);
                if (!optixSphereOptixIR)
                    return 1;

                nvrtcRes = j.nvrtcApi->nvrtcGetOptiXIR(prog, optixSphereOptixIR.get());
                if (nvrtcRes != ::NVRTC_SUCCESS)
                    return 1;

                j.nvrtcApi->nvrtcDestroyProgram(&prog);

                pipelineCompileOptions.usesMotionBlur        = false;
                pipelineCompileOptions.traversableGraphFlags = ::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
                pipelineCompileOptions.numPayloadValues      = 3; // 3 floats, ie a color
                pipelineCompileOptions.numAttributeValues    = 1;
                pipelineCompileOptions.exceptionFlags        = ::OPTIX_EXCEPTION_FLAG_NONE; // no exception enabled
                pipelineCompileOptions.pipelineLaunchParamsVariableName = "params"; // extern "C" __constant__ variable
                pipelineCompileOptions.usesPrimitiveTypeFlags           = ::OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

                // first two are used only in debug
                OptixModuleCompileOptions moduleCompileOptions{};
                moduleCompileOptions.optLevel   = ::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                moduleCompileOptions.debugLevel = ::OPTIX_COMPILE_DEBUG_LEVEL_FULL;

                logStringLen = logStringSize;
                if (!dmt::optixCall(optixModuleCreate(j.optixContext,
                                                      &moduleCompileOptions,
                                                      &pipelineCompileOptions,
                                                      optixSphereOptixIR.get(),
                                                      optixIRSize,
                                                      logString.get(),
                                                      &logStringLen,
                                                      &sphereModule)))
                    return 1;

                OptixBuiltinISOptions builtinISoptions{};
                builtinISoptions.usesMotionBlur      = false;
                builtinISoptions.builtinISModuleType = ::OPTIX_PRIMITIVE_TYPE_SPHERE;
                if (!dmt::optixCall(optixBuiltinISModuleGet(j.optixContext,
                                                            &moduleCompileOptions,
                                                            &pipelineCompileOptions,
                                                            &builtinISoptions,
                                                            &sphereIntersectionModule)))
                    return 1;
            }

            // -- 3: Create Necessary Program Groups for each program type --
            OptixProgramGroup raygenProgGroup   = nullptr;
            OptixProgramGroup missProgGroup     = nullptr;
            OptixProgramGroup hitgroupProgGroup = nullptr;
            {
                // for each program group, define a Group Descriptor and call `optixProgramGroupCreate`
                OptixProgramGroupOptions programGroupOptions{};

                OptixProgramGroupDesc raygenProgGroupDesc{};
                raygenProgGroupDesc.kind                     = ::OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygenProgGroupDesc.raygen.module            = sphereModule;
                raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__rg";

                logStringLen = logStringSize;
                if (!dmt::optixCall(optixProgramGroupCreate(j.optixContext,
                                                            &raygenProgGroupDesc,
                                                            1 /*num program groups*/,
                                                            &programGroupOptions,
                                                            logString.get(),
                                                            &logStringLen,
                                                            &raygenProgGroup)))
                    return 1;

                OptixProgramGroupDesc missProgGroupDesc{};
                missProgGroupDesc.kind                   = ::OPTIX_PROGRAM_GROUP_KIND_MISS;
                missProgGroupDesc.miss.module            = sphereModule;
                missProgGroupDesc.miss.entryFunctionName = "__miss__ms";

                logStringLen = logStringSize;
                if (!dmt::optixCall(optixProgramGroupCreate(j.optixContext,
                                                            &missProgGroupDesc,
                                                            1 /*num program groups*/,
                                                            &programGroupOptions,
                                                            logString.get(),
                                                            &logStringLen,
                                                            &missProgGroup)))
                    return 1;

                OptixProgramGroupDesc hitgroupProgGroupDesc{};
                hitgroupProgGroupDesc.kind                         = ::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hitgroupProgGroupDesc.hitgroup.moduleCH            = sphereModule;
                hitgroupProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
                hitgroupProgGroupDesc.hitgroup.moduleAH            = nullptr;
                hitgroupProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;
                hitgroupProgGroupDesc.hitgroup.moduleIS            = sphereIntersectionModule;
                hitgroupProgGroupDesc.hitgroup.entryFunctionNameIS = nullptr;

                logStringLen = logStringSize;
                if (!dmt::optixCall(optixProgramGroupCreate(j.optixContext,
                                                            &hitgroupProgGroupDesc,
                                                            1 /*num program groups*/,
                                                            &programGroupOptions,
                                                            logString.get(),
                                                            &logStringLen,
                                                            &hitgroupProgGroup)))
                    return 1;
            }

            // -- 4: Link Pipeline: optixPipelineCreate plus set stack size --
            OptixPipeline pipeline = nullptr;
            {
                uint32_t const            maxTraceDepth           = 1;
                static constexpr uint32_t programGroupsLen        = 3;
                OptixProgramGroup programGroups[programGroupsLen] = {raygenProgGroup, missProgGroup, hitgroupProgGroup};

                OptixPipelineLinkOptions pipelineLinkOptions{};
                pipelineLinkOptions.maxTraceDepth = maxTraceDepth;

                logStringLen = logStringSize;
                if (!dmt::optixCall(optixPipelineCreate(j.optixContext,
                                                        &pipelineCompileOptions,
                                                        &pipelineLinkOptions,
                                                        programGroups,
                                                        programGroupsLen,
                                                        logString.get(),
                                                        &logStringLen,
                                                        &pipeline)))
                    return 1;

                // Now set the optimal stack size for the pipeline programs (`optix_stack_size.h`)
                OptixStackSizes stackSizes{};
                for (auto const& progGroup : programGroups)
                {
                    if (!dmt::optixCall(optixUtilAccumulateStackSizes(progGroup, &stackSizes, pipeline)))
                        return 1;
                }

                uint32_t directCallableStackSizeFromTraversal = 0;
                uint32_t directCallableStackSizeFromState     = 0;
                uint32_t continuationStackSize                = 0;

                if (!dmt::optixCall(optixUtilComputeStackSizes(&stackSizes,
                                                               maxTraceDepth,
                                                               0 /*maxCCDepth*/,
                                                               0 /*maxDCDepth*/,
                                                               &directCallableStackSizeFromTraversal,
                                                               &directCallableStackSizeFromState,
                                                               &continuationStackSize)))
                    return 1;

                if (!dmt::optixCall(optixPipelineSetStackSize(pipeline,
                                                              directCallableStackSizeFromTraversal,
                                                              directCallableStackSizeFromState,
                                                              continuationStackSize,
                                                              1 /*maxTraversableDepth*/)))
                    return 1;
            }

            // -- 5: Set up shader binding table --
            OptixShaderBindingTable sbt{};
            {
                // memory leak!
                CUdeviceptr  raygenRecord     = 0;
                size_t const raygenRecordSize = sizeof(RayGenSbtRecord);
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&raygenRecord, raygenRecordSize)))
                    return 1;

                Camera cam;
                configureCamera(cam, width, height);

                RayGenSbtRecord rgSbt{};
                memcpy(reinterpret_cast<float*>(&rgSbt.data.cam_eye), cam.eye, 3 * sizeof(float));
                cam.uvwFrame(reinterpret_cast<float*>(&rgSbt.data.camera_u),
                             reinterpret_cast<float*>(&rgSbt.data.camera_v),
                             reinterpret_cast<float*>(&rgSbt.data.camera_w));

                if (!dmt::optixCall(optixSbtRecordPackHeader(raygenProgGroup, &rgSbt)))
                    return 1;
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemcpyHtoD(raygenRecord, &rgSbt, raygenRecordSize)))
                    return 1;

                CUdeviceptr  missRecord     = 0;
                size_t const missRecordSize = sizeof(MissSbtRecord);
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&missRecord, missRecordSize)))
                    return 1;

                MissSbtRecord missSbt{};
                missSbt.data.r = 0.3f, missSbt.data.g = 0.1f, missSbt.data.b = 0.2f;

                if (!dmt::optixCall(optixSbtRecordPackHeader(missProgGroup, &missSbt)))
                    return 1;
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemcpyHtoD(missRecord, &missSbt, missRecordSize)))
                    return 1;

                CUdeviceptr  hitgroupRecord     = 0;
                size_t const hitgroupRecordSize = sizeof(HitGroupSbtRecord);
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&hitgroupRecord, hitgroupRecordSize)))
                    return 1;

                HitGroupSbtRecord hgSbt{};

                if (!dmt::optixCall(optixSbtRecordPackHeader(hitgroupProgGroup, &hgSbt)))
                    return 1;
                if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                         j.cudaApi->cuMemcpyHtoD(hitgroupRecord, &hgSbt, hitgroupRecordSize)))
                    return 1;

                // now set up all device pointers inside the shader binding table
                sbt.raygenRecord                = raygenRecord;
                sbt.missRecordBase              = missRecord;
                sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
                sbt.missRecordCount             = 1;
                sbt.hitgroupRecordBase          = hitgroupRecord;
                sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
                sbt.hitgroupRecordCount         = 1;
            }

            // -- 6: prepare output buffer --
            CUdeviceptr                     d_outputBuffer = 0;
            dmt::UniqueRef<unsigned char[]> outputBuffer   = nullptr;
            static uint32_t constexpr numChannels = 4;
            uint32_t const outputNumBytes = width * height * numChannels * sizeof(unsigned char);
            {
                if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                         j.cudaApi->cuMemAlloc(&d_outputBuffer, outputNumBytes)))
                    return 1;

                outputBuffer = dmt::makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), numChannels * width * height);
                if (!outputBuffer)
                {
                    ctx.error("Couldn't allocate host memory for output image", {});
                    return 1;
                }
            }

            // -- 7: launch --
            {
                CUstream stream = 0;
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuStreamCreate(&stream, 0)))
                    return 1;

                Params params{};
                params.image        = reinterpret_cast<decltype(params.image)>(d_outputBuffer);
                params.image_width  = width;
                params.image_height = height;
                params.origin_x     = width / 2;
                params.origin_y     = height / 2;
                params.handle       = gasHandle;

                CUdeviceptr d_params = 0;
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemAlloc(&d_params, sizeof(Params))))
                    return 1;
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemcpyHtoD(d_params, &params, sizeof(Params))))
                    return 1;

                ctx.log("Launching Optix Kernel", {});
                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxSynchronize()))
                    return 1;

                if (!dmt::optixCall(
                        optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height, /*depth*/ 1)))
                {
                    CUresult cuErr = j.cudaApi->cuStreamSynchronize(stream);
                    if (cuErr != CUDA_SUCCESS)
                    {
                        char const* name = nullptr;
                        char const* desc = nullptr;
                        j.cudaApi->cuGetErrorName(cuErr, &name);
                        j.cudaApi->cuGetErrorString(cuErr, &desc);
                        ctx.error("CUDA Driver API Error: {} - {}",
                                  std::make_tuple(std::string_view(name), std::string_view{desc}));
                    }
                    return 1;
                }

                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuStreamSynchronize(stream)))
                    return 1;

                ctx.log("Optix Kernel Completed", {});
                j.cudaApi->cuMemFree(d_params);
                j.cudaApi->cuStreamDestroy(stream);

                if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemcpyDtoH(outputBuffer.get(), d_outputBuffer, outputNumBytes)))
                    return 1;

                dmt::os::Path imagePath    = dmt::os::Path::executableDir() / "image.png";
                auto          imagePathStr = imagePath.toUnderlying();
                ctx.log("Saving image to {}", std::make_tuple(imagePathStr));
                printf("Saving image to %s", imagePathStr.c_str());
                if (!stbi_write_png(imagePathStr.c_str(), width, height, numChannels, outputBuffer.get(), width * numChannels))
                {
                    ctx.error("Couldn't save image", {});
                    return 1;
                }
            }

            // -- 8: cleanup --
            {
                j.cudaApi->cuMemFree(sbt.raygenRecord);
                j.cudaApi->cuMemFree(sbt.missRecordBase);
                j.cudaApi->cuMemFree(sbt.hitgroupRecordBase);

                optixPipelineDestroy(pipeline);
                optixProgramGroupDestroy(hitgroupProgGroup);
                optixProgramGroupDestroy(missProgGroup);
                optixProgramGroupDestroy(raygenProgGroup);
                optixModuleDestroy(sphereModule);
                optixModuleDestroy(sphereIntersectionModule);
            }
        }
    }

    return 0;
}