#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "platform/platform-logging-default-formatters.h"

// testing cuda wrapper
#define DMT_CUDAUTILS_IMPL
#include "platform/cuda-wrapper.h"
#include "cudautils/cudautils.h"
#include "cuda-wrappers/cuda-nvrtc.h"
#include "cuda-wrappers/cuda-runtime.h"

// windwoos only
#if defined(_WIN32)
#include <Psapi.h>
#include <iostream>
#endif

#if defined(_WIN32)
static uint32_t utf8FromUtf16le(wchar_t const* wideStr, char* output)
{
    if (!wideStr)
        return 0;

    // Get required buffer size
    DWORD utf8Size = WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, NULL, 0, NULL, NULL);
    if (output == nullptr)
        return utf8Size;
    else // Convert to UTF-8
    {
        if (WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, output, utf8Size, NULL, NULL) == 0)
            return 0;

        return 0;
    }
}
#endif


namespace dmt {
    void listLoadedDLLs()
    {
#if defined(_WIN32)
        HANDLE       hProcess = GetCurrentProcess();
        HMODULE      hMods[1024];
        DWORD        needed = 0;
        dmt::Context ctx;

        if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &needed))
        {
            wchar_t* modName  = new wchar_t[1024];
            char*    uModName = new char[1024];
            for (int32_t i = 1 /*skip .exe*/; i < (needed / sizeof(HMODULE)); ++i)
            {
                if (DWORD len = GetModuleFileNameExW(hProcess, hMods[i], modName, 1024); len > 0)
                {
                    utf8FromUtf16le(modName, uModName);
                    ctx.log("Loaded DLL: {}", std::make_tuple(uModName));
                }
            }
            delete[] modName;
            delete[] uModName;
        }
#endif
    }

    // TODO move elsewhere
    [[noreturn]] void cudaDriverCall(NvcudaLibraryFunctions* cudaApi, CUresult result) 
    {
        if (result == ::CUDA_SUCCESS)
            return;

        Context     ctx;
        char const* errorStr = nullptr;
        cudaApi->cuGetErrorString(result, &errorStr);
        ctx.error("Couln't get the device. Error: {}", std::make_tuple(errorStr));
        exit(1);
    }

    // TODO Move elsewhere. This function (and similiar for all loaded dlls) should be populated with more
    // manual fixes with respect to the generated version as soon as Access Violations are discovered
    void fixCUDADriverSymbols(NvcudaLibraryFunctions* cudaApi)
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

static std::pmr::string const saxpy = R"a(
using int32_t = int;

__device__ int32_t globalThreadIndex()
{
    int32_t const column          = threadIdx.x;
    int32_t const row             = threadIdx.y;
    int32_t const aisle           = threadIdx.z;
    int32_t const threads_per_row = blockDim.x;                  //# threads in x direction aka row
    int32_t const threads_per_aisle = (blockDim.x * blockDim.y); //# threads in x and y direction for total threads per aisle

    int32_t const threads_per_block = (blockDim.x * blockDim.y * blockDim.z);
    int32_t const rowOffset         = (row * threads_per_row);     //how many rows to push out offset by
    int32_t const aisleOffset       = (aisle * threads_per_aisle); // how many aisles to push out offset by

    //S32_t constecond section locates and caculates block offset withing the grid
    int32_t const blockColumn    = blockIdx.x;
    int32_t const blockRow       = blockIdx.y;
    int32_t const blockAisle     = blockIdx.z;
    int32_t const blocks_per_row = gridDim.x;                 //# blocks in x direction aka blocks per row
    int32_t const blocks_per_aisle = (gridDim.x * gridDim.y); // # blocks in x and y direction for total blocks per aisle
    int32_t const blockRowOffset   = (blockRow * blocks_per_row);     // how many rows to push out block offset by
    int32_t const blockAisleOffset = (blockAisle * blocks_per_aisle); // how many aisles to push out block offset by
    int32_t const blockId          = blockColumn + blockRowOffset + blockAisleOffset;

    int32_t const blockOffset = (blockId * threads_per_block);

    int32_t const gid = (blockOffset + aisleOffset + rowOffset + column);
    return gid;
}

extern "C" __global__ void saxpyKernel(int32_t count, float a, float const* x, float const* y, float* result)
{
    int32_t threadId = globalThreadIndex();
    if (threadId < count)
    {
        result[threadId] = a * x[threadId] + y[threadId];
    }
}
)a";

int32_t guardedMain()
{
    using namespace std::string_view_literals;
    auto res = dmt::ctx::addContext(true);
    dmt::ctx::cs->setActive(0);

    dmt::Context ctx;
    ctx.impl()->addHandler([](dmt::LogHandler& _out) { dmt::createConsoleHandler(_out); });
    ctx.log("Hello World", {});
    dmt::listLoadedDLLs();

    ctx.log("Starting Path tests", {});

    // Test 1: Get Executable Directory
    dmt::os::Path exeDir = dmt::os::Path::executableDir();
    ctx.log("Executable Path: {}", std::make_tuple(exeDir.toUnderlying()));

    // Test 2: Get Home Directory
    dmt::os::Path homeDir = dmt::os::Path::home();
    ctx.log("Home Directory: {}", std::make_tuple(homeDir.toUnderlying()));

    // Test 3: Get Current Working Directory
    dmt::os::Path cwd = dmt::os::Path::cwd();
    ctx.log("Current Working Directory: {}", std::make_tuple(cwd.toUnderlying()));

    // Test 4: Root Path from Disk Designator
    dmt::os::Path rootPath = dmt::os::Path::root("C:");
    ctx.log("Root Path: {}\n", std::make_tuple(rootPath.toUnderlying()));

    // Test 5: Parent Directory
    dmt::os::Path parentPath = exeDir.parent();
    ctx.log("Parent of Executable Path: {}", std::make_tuple(parentPath.toUnderlying()));

    // Test 6: Modify Path using Parent_
    dmt::os::Path mutablePath = exeDir;
    mutablePath.parent_();
    ctx.log("After parent_() call: {}", std::make_tuple(mutablePath.toUnderlying()));

    // Test 7: Append Path Component
    dmt::os::Path appendedPath = exeDir / "testFolder";
    ctx.log("Appended Path: {}", std::make_tuple(appendedPath.toUnderlying()));

    // Test 8: Check Validity and Directory/File Status
    ctx.log("Is Executable Directory valid?: {}", std::make_tuple(exeDir.isValid()));
    ctx.log("Is Executable Directory actually a directory?: {}", std::make_tuple(exeDir.isDirectory()));

    // Test 9: Test Move Constructor
    dmt::os::Path movedPath = std::move(exeDir);
    ctx.log("Moved Path: {}", std::make_tuple(movedPath.toUnderlying()));

    // Test 10: Test Copy Constructor
    dmt::os::Path copiedPath = homeDir;
    ctx.log("Copied Path: {}", std::make_tuple(copiedPath.toUnderlying()));

    ctx.log("Path tests completed.", {});
    // continue testing and logging

    // TODO add a library loader inside the context
    dmt::os::LibraryLoader loader{false};

    // Test 11: CUDA Driver API
    std::unique_ptr<NvcudaLibraryFunctions> cudaApi = std::make_unique<NvcudaLibraryFunctions>();
    if (!loadNvcudaFunctions(loader, cudaApi.get()))
    {
        ctx.error("Couldn't load nvcuda.dll", {});
        return 1;
    }
    dmt::fixCUDADriverSymbols(cudaApi.get());

    ctx.log("Loaded cuda library, meaning the list of loaded DLLs should contain nvcuda", {});
    dmt::listLoadedDLLs();
    ctx.log("Loaded cuda library, trying a random function", {});
    CUdevice deviceF;
    dmt::cudaDriverCall(cudaApi.get(), cudaApi->cuInit(0));
    // Testing that the CUDA Driver api actually works
    CUresult    driverRes = cudaApi->cuCtxGetDevice(&deviceF);
    char const* errorStr = nullptr;
    cudaApi->cuGetErrorString(driverRes, &errorStr);
    if (driverRes != ::CUDA_SUCCESS)
        ctx.error("Couln't get the device. Error: {}", std::make_tuple(errorStr));
    else
        ctx.log("Got device", {});

    // Test: CUDA Runtime Library
    std::unique_ptr<Cudart64_12LibraryFunctions> cudartApi = std::make_unique<Cudart64_12LibraryFunctions>();
    if (!loadCudart64_12Functions(loader, cudartApi.get()))
    {
        ctx.error("Couldn't load CUDA Runtime Library", {});
        return 1;
    }

    // Test: CUDA NVRTC Library
    std::unique_ptr<Nvrtc64_120_0LibraryFunctions> nvrtcApi = std::make_unique<Nvrtc64_120_0LibraryFunctions>();
    if (!loadNvrtc64_120_0Functions(loader, nvrtcApi.get()))
    {
        ctx.error("Couldn't load CUDA NVRTC Library", {});
        return 1;
    }

    ctx.log("Listing Loaded DLLs again. Should contain build cudart and nvrtc", {});
    dmt::listLoadedDLLs();

    // New retrieve the first CUDA Capable device (assuming there is one here) and create a context
    CUcontext cuContext;
    CUdevice  device;
    dmt::cudaDriverCall(cudaApi.get(), cudaApi->cuDeviceGet(&device, 0));
    dmt::cudaDriverCall(cudaApi.get(), cudaApi->cuCtxCreate(&cuContext, 0, device));
    ctx.log("CUDA Context initialized successfully", {});

    // Initialize cuda context and device (through function pointers struct)
    nvrtcProgram prog;
    nvrtcResult nvrtcRes = nvrtcApi->nvrtcCreateProgram(&prog, saxpy.c_str(), "saxpyKernel.cu", 0, nullptr, nullptr);
    if (nvrtcRes != ::NVRTC_SUCCESS)
    {
        ctx.error("nvrtc Failed: {}", std::make_tuple(nvrtcApi->nvrtcGetErrorString(nvrtcRes)));
        return 1;
    }

    // TODO: compiler options passed through cmake or generated somehow
    nvrtcRes = nvrtcApi->nvrtcCompileProgram(prog, 0, nullptr);
    if (nvrtcRes != ::NVRTC_SUCCESS)
    {
        size_t logSize;
        nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtcApi->nvrtcGetProgramLog(prog, log.data());
        ctx.error("NVRTC Compilation Failed: {}", std::make_tuple(log));
        return 1;
    }
    ctx.log("NVRTC Compilation Successful", {});

    // use nvrtc to compile and link into ptx (through function pointers struct)
    size_t ptxSize = 0;
    nvrtcApi->nvrtcGetPTXSize(prog, &ptxSize);
    std::unique_ptr<char[]> ptx = std::make_unique<char[]>(ptxSize);
    nvrtcApi->nvrtcGetPTX(prog, ptx.get());
    nvrtcApi->nvrtcDestroyProgram(&prog);

    ctx.log("Successfully created a PTX from CUDA Kernel", {});

    // use driver api to load the compiled function in the current context and get the CUfunction
    CUmodule cuModule;
    CUfunction saxpyFunction;

    dmt::cudaDriverCall(cudaApi.get(), cudaApi->cuModuleLoadData(&cuModule, ptx.get()));
    dmt::cudaDriverCall(cudaApi.get(), cudaApi->cuModuleGetFunction(&saxpyFunction, cuModule, "saxpyKernel"));

    ctx.log("CUDA Kernel loaded successfully", {});

    // allocate some device memory and 
    int32_t count = 1024;
    float   a     = 2.0f;
    std::pmr::vector<float> x(count, 1.f), y(count, 2.f), result(count, 0.f);
    CUdeviceptr             d_x, d_y, d_result;

    cudaApi->cuMemAlloc(&d_x, count * sizeof(float)); // uses device of current context
    cudaApi->cuMemAlloc(&d_y, count * sizeof(float));
    cudaApi->cuMemAlloc(&d_result, count * sizeof(float));

    cudaApi->cuMemcpyHtoD(d_x, x.data(), count * sizeof(float));
    cudaApi->cuMemcpyHtoD(d_y, y.data(), count * sizeof(float));

    // launch the kernel
    void* args[] = { &count, &a, &d_x, &d_y, &d_result };
    int   threadsPerBlock = 256;
    int   blocksPerGrid   = (count + threadsPerBlock - 1) / threadsPerBlock; // ceilDiv(count, threadsPerBlock)

    // clang-format 0ff
    driverRes = cudaApi->cuLaunchKernel(
        saxpyFunction, 
        blocksPerGrid, 1, 1,   // Grid Dimensions
        threadsPerBlock, 1, 1, // Block Dimensions
        0, nullptr,            // shared memory and stream
        args, nullptr
    );
    // clang-format on
    dmt::cudaDriverCall(cudaApi.get(), driverRes);

    cudaApi->cuCtxSynchronize();

    // copy back result and print it
    dmt::cudaDriverCall(cudaApi.get(), cudaApi->cuMemcpyDtoH(result.data(), d_result, count * sizeof(float)));

    ctx.log("Printing first 10 results of Saxpy {} * x + y", std::make_tuple(a));
    for (int32_t i = 0; i < 10; ++i)
        ctx.log("Result[{}]: {}", std::make_tuple(i, result[i]));

    // cleanup
    cudaApi->cuMemFree(d_x);
    cudaApi->cuMemFree(d_y);
    cudaApi->cuMemFree(d_result);
    cudaApi->cuModuleUnload(cuModule);
    cudaApi->cuCtxDestroy(cuContext);

    return 0;
}
