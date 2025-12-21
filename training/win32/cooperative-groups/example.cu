#include "example.h"

// CUDA stuff
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>

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

    // ------------------------------------------------- saxpy --------------------------------------------------------

    // expects a 1 dimensional grid. each thread will access 16 bytes (4 numbers)
    __global__ void ksaxpy(float const* v0, float const* v1, float* v2, float a, uint32_t count)
    {
        namespace cg              = cooperative_groups;
        uint32_t const arrayIndex = cg::grid_group::block_index().x * cg::thread_block::group_index().x;
        if (arrayIndex < count)
        {
            float4 const first  = reinterpret_cast<float4 const*>(v0)[arrayIndex >> 2];
            float4 const second = reinterpret_cast<float4 const*>(v1)[arrayIndex >> 2];
            float4       result = reinterpret_cast<float4*>(v2)[arrayIndex >> 2];

            result.x = a * first.x + second.x;
            result.y = a * first.y + second.y;
            result.z = a * first.z + second.z;
            result.w = a * first.w + second.w;

            memcpy(reinterpret_cast<float4*>(v2) + (arrayIndex >> 2), &result, sizeof(float4));
        }
    }

    void saxpy(float const* v0, float const* v1, float* v2, float a, size_t count)
    {
        void*          devPtr = nullptr;
        uint32_t const uCount = static_cast<uint32_t>(count);
        size_t const   bytes  = count * sizeof(float);
        if (!CUDA_SUCC(cudaMalloc(&devPtr, 3 * count * sizeof(float))))
            return;
        class Janitor
        {
        public:
            explicit Janitor(void* devPtr) : m_thePtr(devPtr) {}
            ~Janitor() { CUDA_SUCC(cudaFree(m_thePtr)); }

        private:
            void* m_thePtr;
        };
        // allocate all the data
        auto* d_first  = reinterpret_cast<float*>(static_cast<unsigned char*>(devPtr) + 0 * bytes);
        auto* d_second = reinterpret_cast<float*>(static_cast<unsigned char*>(devPtr) + 1 * bytes);
        auto* d_result = reinterpret_cast<float*>(static_cast<unsigned char*>(devPtr) + 2 * bytes);
        if (!CUDA_SUCC(cudaMemcpy(d_first, v0, bytes, ::cudaMemcpyHostToDevice)))
            return;
        if (!CUDA_SUCC(cudaMemcpy(d_second, v1, bytes, ::cudaMemcpyHostToDevice)))
            return;
        // now run the kernel and synchronize the whole context
        static constexpr uint32_t elementsPerBlock = 256;

        dim3 const blockDim{elementsPerBlock, 1, 1};
        dim3 const gridDim{(uCount + elementsPerBlock - 1) / elementsPerBlock, 1, 1};
        ksaxpy<<<gridDim, blockDim, 0, 0>>>(d_first, d_second, d_result, a, uCount);

        bool notSucc = !CUDA_SUCC(cudaGetLastError());
        notSucc      = notSucc || !CUDA_SUCC(cudaDeviceSynchronize());
        if (notSucc)
            return;

        CUDA_SUCC(cudaMemcpy(v2, d_result, bytes, ::cudaMemcpyDeviceToHost));
    }

    void matrixSum(float const* A, float const* B, size_t rows, size_t cols, float* result) {}

} // namespace dmt::_1basics