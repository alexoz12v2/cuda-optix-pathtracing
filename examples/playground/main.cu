#include "printf.cuh"

#define DMT_ENTRY_POINT
#include <platform/platform.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
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
        DMT_CPU_GPU inline constexpr void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
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
        DMT_CPU_GPU inline constexpr void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
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

    // TODO pstd::string_view
    template <std::convertible_to<std::string_view> T>
    struct UTF8Formatter<T>
    {
        DMT_CPU_GPU inline constexpr void operator()(T const& value, char8_t* _buffer, uint32_t& _offset, uint32_t& _bufferSize)
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

    std::cout << buffer << std::endl;

    err = cudaFree(buffer);
    assert(err == ::cudaSuccess);
}

static __global__ void kContext()
{
    int32_t gid = globalThreadIndex();
    Context ctx;
    ctx.warn("fdsafdaaf\xf0\x9f\x98\x8a {}", std::make_tuple(3.f));
}

static int32_t s_keyPressed = 0;

static __global__ void kWriteFile(os::CudaFileMapping* pFileMapping, uint32_t chunkSize)
{
    pFileMapping->requestChunk(0, 0);
    uint32_t offset = threadIdx.x * (chunkSize >> 4);
    // TODO: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st
    int* p = std::bit_cast<int*>(std::bit_cast<uintptr_t>(pFileMapping->target) + offset);
    p[0]   = threadIdx.x;
    p[1]   = threadIdx.y;
    pFileMapping->signalCompletion();
}

static void testNewContext()
{
    using namespace std::string_view_literals;
    auto res = ctx::addContext(true);
    if (res != ctx::ECtxReturn::eCreatedOnManaged)
        std::abort();
    ctx::cs->setActive(0);

    // context is available here
    {
        Context ctx;
        ctx.impl()->addHandler([](LogHandler& _out) { createConsoleHandler(_out); });
        // this is equivalent to "fdsafdaaf\xf0\x9f\x98\x8a {}", but NOT u8"fdsafdaaf\xf0\x9f\x98\x8a {}"
        //static constexpr char8_t fmtstr[] = {u8'f', u8'd', u8's', u8'a', u8'f', u8'd', u8's', u8'a', u8'f', 0xF0, 0x9F, 0x98, 0x8A, u8'{', u8'}'};
        ctx.warn("fdsafdaaf\xf0\x9f\x98\x8a {}", std::make_tuple(3.f));
        ctx.error("fdsafdaaf\xf0\x9f\x98\x8a {}", std::make_tuple(3.f));
        ctx.log("fdsafdaaf\xf0\x9f\x98\x8a {}", std::make_tuple(3.f));
        ctx.trace("fdsafdaaf\xf0\x9f\x98\x8a {}", std::make_tuple(3.f));
        ctx.flush();
        kContext<<<1, 32>>>();
        cudaError_t err = cudaDeviceSynchronize();
        if (err != ::cudaSuccess)
            ctx.error("Failed context kernel execution \xF0\x9F\x99\x81", std::make_tuple());

        static constexpr uint32_t _64KB  = 64 * 1024;
        void*                     memory = nullptr;
        cudaMalloc(&memory, _64KB); // LEAKED

        os::CudaFileMapping* pFileMapping = nullptr;

        err = cudaMallocManaged(&pFileMapping, sizeof(os::CudaFileMapping));
        if (err != ::cudaSuccess)
            std::abort();
        std::construct_at(pFileMapping, "bonk.txt"sv, _64KB, true, memory);
        std::atomic<int32_t> stopRequested = 2;

        ctx.log("roba {}", std::make_tuple(*pFileMapping->chunky()));

        auto const iothreadEntrypoint = [](os::CudaFileMapping* _pFileMapping, std::atomic<int32_t>* _pStopRequested) {
            while (_pStopRequested->load(std::memory_order_acquire) != 0)
                std::this_thread::yield();
            while (_pStopRequested->load(std::memory_order_acquire) != 1)
            {
                if (_pFileMapping->requestedChunk())
                {
                    // TODO once requested, use janitor
                    _pFileMapping->signalChunkLoaded();
                    _pFileMapping->waitForCompletion();
                }
            }
        };

        // TODO: The kernel crashes. fix it
        std::thread t{iothreadEntrypoint, pFileMapping, &stopRequested};
        kWriteFile<<<1, 32>>>(pFileMapping, _64KB);
        err = cudaGetLastError();
        if (err != ::cudaSuccess)
            __debugbreak();
        stopRequested.store(0, std::memory_order_release);

        while (!s_keyPressed) // hang
        {
        }

        stopRequested.store(1, std::memory_order_release);
        cudaDeviceSynchronize();
        t.join();
        std::destroy_at(pFileMapping);
        cudaFree(pFileMapping);
        cudaFree(memory);
    }
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
        s_keyPressed = 1;
}

class WindowJanitor
{
public:
    WindowJanitor(GLFWkeyfun _keyCallback)
    {
        if (!glfwInit())
            std::abort();

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 1);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // offscreen context
        m_window = glfwCreateWindow(640, 480, "Playground", nullptr, nullptr);
        if (!m_window)
            std::abort();
        glfwMakeContextCurrent(m_window);
        glfwSetKeyCallback(m_window, _keyCallback);
    }

    WindowJanitor(WindowJanitor const&)                = delete;
    WindowJanitor(WindowJanitor&&) noexcept            = delete;
    WindowJanitor& operator=(WindowJanitor const&)     = delete;
    WindowJanitor& operator=(WindowJanitor&&) noexcept = delete;
    ~WindowJanitor()
    {
        if (m_window)
            glfwDestroyWindow(m_window);
        glfwTerminate();
    }

private:
    GLFWwindow* m_window = nullptr;
};

int guardedMain()
{
    //std::unique_ptr<char8_t[]> ptr    = std::make_unique<char8_t[]>(2048);
    //std::unique_ptr<char8_t[]> args   = std::make_unique<char8_t[]>(2048);
    //auto                       record = createRecord(u8"afdsf {} {}",
    //                           ELogLevel::LOG,
    //                           ptr.get(),
    //                           2048,
    //                           args.get(),
    //                           2048,
    //                           std::make_tuple(3u, 3.f),
    //                           getPhysicalLocation(),
    //                           std::source_location::current());

    //std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    //std::string                                            view{std::bit_cast<char*>(record.data), record.numBytes};
    //std::wstring                                           wstr = converter.from_bytes(view);
    //std::wcout << wstr << std::endl;

    auto v = os::getEnv();
    dmt::cudaHello(nullptr);
    WindowJanitor wj{keyCallback};

    //testDevicePrint();
    testNewContext();
    return 0;
}