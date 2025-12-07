#include "example.h"

// CUDA stuff
#include <cuda_runtime.h>

// std
#include <iostream>

// windows
#include <Windows.h>

namespace dmt::win32 {
    [[noreturn]] void printResultAndExitProcess(HRESULT result)
    {
        wchar_t* errorText = nullptr;
        if (FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_FROM_SYSTEM,
                           nullptr,
                           result,
                           MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                           reinterpret_cast<LPWSTR>(&errorText),
                           0,
                           nullptr))
        {
            std::wcout << ANSI_RED << errorText << ANSI_RST << std::endl;
            LocalFree(errorText);
        }
        else
        {
            std::cout << ANSI_RED "Unrecognized Win32 Error" ANSI_RST << std::endl;
        }
        ExitProcess(1);
    }
} // namespace dmt::win32

namespace dmt::_1basics {

    void printCudaCapableDevices()
    {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != ::cudaSuccess)
        {
            cudaError_t const err = cudaGetLastError();
            std::cerr << ANSI_RED "[" << cudaGetErrorName(err) << "]: " << cudaGetErrorString(err) << ANSI_RST << std::endl;
            return;
        }

        std::cout << "Enumerated CUDA Capable devices [" << deviceCount << "]" << std::endl;
    }

    void matrixSum(float const* A, float const* B, size_t rows, size_t cols, float* result) {}

} // namespace dmt::_1basics
