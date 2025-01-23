#include "printf.cuh"

#include <platform/platform-logging.h>
#include <platform/platform-cuda-utils.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <memory>
#include <iostream>
#include <tuple>
#include <locale>
#include <codecvt>
#include <string>
#include <string_view>
#include <source_location>

#include <cstdint>
#include <cstdlib>

namespace dmt {
    template <std::floating_point T>
    struct UTF8Formatter<T>
    {
        DMT_CPU_GPU constexpr inline void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            char* writePos = reinterpret_cast<char*>(_buffer + _offset + 2 * sizeof(uint32_t));
#if defined(__CUDA_ARCH__)
            int bytesWritten = ::dmt::snprintf(writePos,
                                               _bufferSize - _offset - 2 * sizeof(uint32_t),
                                               "%.6g",
                                               value); // Adjust precision if needed
#else
            int bytesWritten = std::snprintf(writePos, _bufferSize - _offset - 2 * sizeof(uint32_t), "%.6g", value); // Adjust precision if needed
#endif

            if (bytesWritten > 0 && static_cast<uint32_t>(bytesWritten) <= (_bufferSize - _offset - 2 * sizeof(uint32_t)))
            {
                uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
                uint32_t len      = numBytes; // Each byte corresponds to one character in this case

                *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
                *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

                _offset += 2 * sizeof(uint32_t) + numBytes;
                _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
            }
            else
            {
                _bufferSize = 0; // Insufficient buffer space
            }
        }
    };

    template <std::integral T>
    struct UTF8Formatter<T>
    {
        DMT_CPU_GPU constexpr inline void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            char* writePos = reinterpret_cast<char*>(_buffer + _offset + 2 * sizeof(uint32_t));
#if defined(__CUDA_ARCH__)
            int bytesWritten = ::dmt::snprintf(writePos, _bufferSize - _offset - 2 * sizeof(uint32_t), "%d", value);
#else
            int bytesWritten = std::snprintf(writePos, _bufferSize - _offset - 2 * sizeof(uint32_t), "%d", value);
#endif

            if (bytesWritten > 0 && static_cast<uint32_t>(bytesWritten) <= (_bufferSize - _offset - 2 * sizeof(uint32_t)))
            {
                uint32_t numBytes = static_cast<uint32_t>(bytesWritten);
                uint32_t len      = numBytes; // Each byte corresponds to one character in this case

                *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
                *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

                _offset += 2 * sizeof(uint32_t) + numBytes;
                _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
            }
            else
            {
                _bufferSize = 0; // Insufficient buffer space
            }
        }
    };

    template <std::convertible_to<std::string_view> T>
    struct UTF8Formatter<T>
    {
        DMT_CPU_GPU constexpr inline void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
        {
            std::string_view strView = value;

            if (_bufferSize < _offset + 2 * sizeof(uint32_t))
                return; // Not enough space for metadata

            uint32_t numBytes = static_cast<uint32_t>(strView.size());
            uint32_t len = static_cast<uint32_t>(strView.size()); // Assuming valid UTF-8 input where each character is 1 byte

            if (_bufferSize < _offset + 2 * sizeof(uint32_t) + numBytes)
                return; // Insufficient space for the string

            *std::bit_cast<uint32_t*>(_buffer + _offset)                    = numBytes; // Store `numBytes`
            *std::bit_cast<uint32_t*>(_buffer + _offset + sizeof(uint32_t)) = len;      // Store `len`

            memcpy(_buffer + _offset + 2 * sizeof(uint32_t), strView.data(), numBytes);

            _offset += 2 * sizeof(uint32_t) + numBytes;
            _bufferSize -= 2 * sizeof(uint32_t) + numBytes;
        }
    };


} // namespace dmt

using namespace dmt;

static __global__ void kPrint(char* buffer, int32_t bufferSize)
{
    int32_t gid = globalThreadIndex();
    if (gid == 0)
    {
        ::dmt::snprintf(buffer, bufferSize, "Hello From Device gid = %d\n", gid);
    }
    __syncthreads();
}

static void testDevicePrint()
{
    char*         buffer;
    int32_t const bufferSize = 4096;
    cudaError_t   err        = cudaMallocManaged(&buffer, bufferSize);
    assert(err == ::cudaSuccess);

    kPrint<<<1, 32>>>(buffer, bufferSize);
    err = cudaGetLastError();
    assert(err == ::cudaSuccess);
    err = cudaDeviceSynchronize();
    assert(err == ::cudaSuccess);

    err = cudaFree(buffer);
    assert(err == ::cudaSuccess);
}

int main()
{
    std::unique_ptr<char8_t[]> ptr  = std::make_unique<char8_t[]>(2048);
    std::unique_ptr<char8_t[]> args = std::make_unique<char8_t[]>(2048);
    auto record = createRecord(u8"afdsf {} {}", ELogLevel::LOG, ptr.get(), 2048, args.get(), 2048, std::make_tuple(3u, 3.f));

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::string                                            view{std::bit_cast<char*>(record.data), record.numBytes};
    std::wstring                                           wstr = converter.from_bytes(view);
    std::wcout << wstr << std::endl;

    testDevicePrint();
}