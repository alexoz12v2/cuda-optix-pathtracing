// std stuff
#include <iostream>

// Windows Stuff
#include "Windows.h"
#include "ShlObj.h"
#include "Objbase.h"

// cuda stuff
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// our stuff
#include "the-macros.h"
#include "example.h"

using namespace dmt;

/// ## About CUDA Runtime and Context Management
/// A [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)
/// is the CUDA Equivalent of a process. All resources and modules allocated for the process are associated to a
/// context.
/// An Implicit Context will be created for each device called Primary Context whenever you call `cudaInitDevice`

int wmain()
{
    // - Setup console properly such that ANSI escape codes work
    for (HANDLE out : {GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE)})
    {
        DWORD mode = 0;
        GetConsoleMode(out, &mode);
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        mode |= DISABLE_NEWLINE_AUTO_RETURN;
        SetConsoleMode(out, mode);
    }
    std::ios::sync_with_stdio();

    // - Print some colored stuff
    std::cout << ANSI_RED "Hello Beautiful World" ANSI_RST << std::endl;

    // initialize COM Apartment for this process
    HRESULT const res = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (!SUCCEEDED(res))
        win32::printResultAndExitProcess(res);

    _1basics::printCudaCapableDevices();
}
