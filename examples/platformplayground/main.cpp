#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "platform/platform-logging-default-formatters.h"

// testing cuda wrapper
#define DMT_CUDAUTILS_IMPL
#include "platform/cuda-wrapper.h"
#include "cudautils/cudautils.h"

// windwoos only
#include <Psapi.h>
#include <iostream>

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
        // TODO better
        if (WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, output, utf8Size, NULL, NULL) == 0)
            return 0;

        return 0;
    }
}


namespace dmt {
    void listLoadedDLLs()
    {
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
    }
} // namespace dmt

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

    std::unique_ptr<NvcudaLibraryFunctions> cudaApi = std::make_unique<NvcudaLibraryFunctions>();
    if (!loadNvcudaFunctions(loader, cudaApi.get()))
    {
        ctx.error("Couldn't load nvcuda.dll", {});
        return 1;
    }

    ctx.log("Loaded cuda library, meaning the list of loaded DLLs should contain nvcuda", {});
    dmt::listLoadedDLLs();
    ctx.log("Loaded cuda library, trying a random function", {});
    CUdevice device;
    cudaApi->cuInit(0);
    CUresult    result   = cudaApi->cuCtxGetDevice(&device);
    char const* errorStr = nullptr;
    cudaApi->cuGetErrorString(result, &errorStr);
    if (result != ::CUDA_SUCCESS)
        ctx.error("Couln't get the device. Error: {}", std::make_tuple(errorStr));
    else
        ctx.log("Got device", {});

    return 0;
}
