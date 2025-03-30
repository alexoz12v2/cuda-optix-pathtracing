#define DMT_ENTRY_POINT
#include "platform/platform.h"

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
    dmt::Ctx::init();
    struct Janitor
    {
        ~Janitor()
        {
            if (cudaApi)
            {
                if (d_x)
                    cudaApi->cuMemFree(d_x);
                if (d_y)
                    cudaApi->cuMemFree(d_y);
                if (d_result)
                    cudaApi->cuMemFree(d_result);
                if (cuModule)
                    cudaApi->cuModuleUnload(cuModule);
                if (cuContext)
                    cudaApi->cuCtxDestroy(cuContext);
            }

            if (nvrtcApi && nvrtcApi->m_library)
                loader.unloadLibrary(nvrtcApi->m_library);
            if (cudartApi && cudartApi->m_library)
                loader.unloadLibrary(cudartApi->m_library);
            if (cudaApi && cudaApi->m_library)
                loader.unloadLibrary(cudaApi->m_library);

            dmt::Ctx::destroy();
        }

        dmt::os::LibraryLoader                         loader{false};
        std::unique_ptr<NvcudaLibraryFunctions>        cudaApi   = nullptr;
        std::unique_ptr<Cudart64_12LibraryFunctions>   cudartApi = nullptr;
        std::unique_ptr<Nvrtc64_120_0LibraryFunctions> nvrtcApi  = nullptr;
        CUcontext                                      cuContext = nullptr;
        CUdeviceptr                                    d_x = 0, d_y = 0, d_result = 0;
        CUmodule                                       cuModule = nullptr;
    } j;

    j.cudaApi   = std::make_unique<NvcudaLibraryFunctions>();
    j.cudartApi = std::make_unique<Cudart64_12LibraryFunctions>();
    j.nvrtcApi  = std::make_unique<Nvrtc64_120_0LibraryFunctions>();

    // Ctx::destroy should be called without any contexts
    {
        dmt::Context ctx;
        if (!ctx.isValid())
            return 1;

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

        // Test 11: CUDA Driver API
        // TODO: These functions are not calling unloadLibrary on the loader.
        if (!loadNvcudaFunctions(j.loader, j.cudaApi.get()))
        {
            ctx.error("Couldn't load nvcuda.dll", {});
            return 1;
        }
        dmt::fixCUDADriverSymbols(j.cudaApi.get());

        ctx.log("Loaded cuda library, meaning the list of loaded DLLs should contain nvcuda", {});
        dmt::listLoadedDLLs();
        ctx.log("Loaded cuda library, trying a random function", {});
        {
            CUdevice deviceF;

            if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuInit(0)))
                return 1;

            // Testing that the CUDA Driver api actually works
            CUresult    driverRes = j.cudaApi->cuCtxGetDevice(&deviceF);
            char const* errorStr  = nullptr;
            j.cudaApi->cuGetErrorString(driverRes, &errorStr);
            if (driverRes != ::CUDA_SUCCESS)
                ctx.error("Couln't get the device. Error: {}", std::make_tuple(errorStr));
            else
                ctx.log("Got device", {});
        }

        // Test: CUDA Runtime Library
        if (!loadCudart64_12Functions(j.loader, j.cudartApi.get()))
        {
            ctx.error("Couldn't load CUDA Runtime Library", {});
            return 1;
        }

        // Test: CUDA NVRTC Library
        if (!loadNvrtc64_120_0Functions(j.loader, j.nvrtcApi.get()))
        {
            ctx.error("Couldn't load CUDA NVRTC Library", {});
            return 1;
        }

        ctx.log("Listing Loaded DLLs again. Should contain build cudart and nvrtc", {});
        dmt::listLoadedDLLs();

        // New retrieve the first CUDA Capable device (assuming there is one here) and create a context
        CUdevice device;
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuDeviceGet(&device, 0)))
            return 1;
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuCtxCreate(&j.cuContext, 0, device)))
            return 1;

        ctx.log("CUDA Context initialized successfully", {});

        // Initialize cuda context and device (through function pointers struct)
        nvrtcProgram prog;
        nvrtcResult nvrtcRes = j.nvrtcApi->nvrtcCreateProgram(&prog, saxpy.c_str(), "saxpyKernel.cu", 0, nullptr, nullptr);
        if (nvrtcRes != ::NVRTC_SUCCESS)
        {
            ctx.error("nvrtc Failed: {}", std::make_tuple(j.nvrtcApi->nvrtcGetErrorString(nvrtcRes)));
            return 1;
        }

        // TODO: compiler options passed through cmake or generated somehow
        nvrtcRes = j.nvrtcApi->nvrtcCompileProgram(prog, 0, nullptr);
        if (nvrtcRes != ::NVRTC_SUCCESS)
        {
            size_t logSize;
            j.nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            j.nvrtcApi->nvrtcGetProgramLog(prog, log.data());
            ctx.error("NVRTC Compilation Failed: {}", std::make_tuple(log));
            return 1;
        }
        ctx.log("NVRTC Compilation Successful", {});

        // use nvrtc to compile and link into ptx (through function pointers struct)
        size_t ptxSize = 0;
        j.nvrtcApi->nvrtcGetPTXSize(prog, &ptxSize);
        std::unique_ptr<char[]> ptx = std::make_unique<char[]>(ptxSize);
        j.nvrtcApi->nvrtcGetPTX(prog, ptx.get());
        j.nvrtcApi->nvrtcDestroyProgram(&prog);

        ctx.log("Successfully created a PTX from CUDA Kernel", {});

        // use driver api to load the compiled function in the current context and get the CUfunction
        CUfunction saxpyFunction;

        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuModuleLoadData(&j.cuModule, ptx.get())))
            return 1;
        if (!dmt::cudaDriverCall(j.cudaApi.get(),
                                 j.cudaApi->cuModuleGetFunction(&saxpyFunction, j.cuModule, "saxpyKernel")))
            return 1;

        ctx.log("CUDA Kernel loaded successfully", {});

        // allocate some device memory and
        int32_t                 count = 1024;
        float                   a     = 2.0f;
        std::pmr::vector<float> x(count, 1.f), y(count, 2.f), result(count, 0.f);

        j.cudaApi->cuMemAlloc(&j.d_x, count * sizeof(float)); // uses device of current context
        j.cudaApi->cuMemAlloc(&j.d_y, count * sizeof(float));
        j.cudaApi->cuMemAlloc(&j.d_result, count * sizeof(float));

        j.cudaApi->cuMemcpyHtoD(j.d_x, x.data(), count * sizeof(float));
        j.cudaApi->cuMemcpyHtoD(j.d_y, y.data(), count * sizeof(float));

        // launch the kernel
        void* args[]          = {&count, &a, &j.d_x, &j.d_y, &j.d_result};
        int   threadsPerBlock = 256;
        int   blocksPerGrid   = (count + threadsPerBlock - 1) / threadsPerBlock; // ceilDiv(count, threadsPerBlock)

        // clang-format off
        CUresult driverRes = j.cudaApi->cuLaunchKernel(
            saxpyFunction,
            blocksPerGrid, 1, 1, // Grid Dimensions
            threadsPerBlock, 1, 1, // Block Dimensions
            0, nullptr, // shared memory and stream
            args, nullptr); // arguments and extra configuration
        // clang-format on
        if (!dmt::cudaDriverCall(j.cudaApi.get(), driverRes))
            return 1;

        j.cudaApi->cuCtxSynchronize();

        // copy back result and print it
        if (!dmt::cudaDriverCall(j.cudaApi.get(), j.cudaApi->cuMemcpyDtoH(result.data(), j.d_result, count * sizeof(float))))
            return 1;

        ctx.log("Printing first 10 results of Saxpy {} * x + y", std::make_tuple(a));
        for (int32_t i = 0; i < 10; ++i)
            ctx.log("Result[{}]: {}", std::make_tuple(i, result[i]));

        // cleanup handled by janitor class (hence do NOT use `exit()`, `abort()`, `std::terminate()`, `ExitProcess`, and such)
    }

    return 0;
}
