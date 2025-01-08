#include "cuda-tests.h"

#define DMT_INTERFACE_AS_HEADER
#include "dmtmacros.h"
#include "platform/platform-cuda-utils.cuh"
#include "platform/platform.h"

#include <cuda_runtime.h>

__global__ void fillBufferKernel(uint8_t* buffer, size_t size, uint8_t value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        buffer[idx] = value;
}


void testBuddyDirectly(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemRes)
{
    using namespace std::string_view_literals;
    dmt::BuddyMemoryResource* pBuddy = dynamic_cast<dmt::BuddyMemoryResource*>(pMemRes);
    assert(pBuddy && "incorrect allocator type");
    actx.error("THis is an error {}", {"err"sv});

    cudaError_t err = cudaGetLastError();
    assert(err == ::cudaSuccess && "the start is already promising...");

    // tests ...
    // === Test 1: Basic Allocation ===
    {
        constexpr size_t blockSize = 256; // Assume a block size for testing
        void*            ptr1      = pBuddy->allocate(blockSize, alignof(std::max_align_t));
        assert(ptr1 != nullptr && "Allocation failed for basic block size");

        void* ptr2 = pBuddy->allocate(blockSize, alignof(std::max_align_t));
        assert(ptr2 != nullptr && ptr1 != ptr2 && "Allocation failed or returned duplicate pointer");

        pBuddy->deallocate(ptr1, blockSize, alignof(std::max_align_t));
        pBuddy->deallocate(ptr2, blockSize, alignof(std::max_align_t));
    }

    // === Test 2: Large Allocation (Edge Case) ===
    {
        size_t const largeBlockSize = pBuddy->maxBlockSize();
        void*        ptr            = pBuddy->allocate(largeBlockSize, alignof(std::max_align_t));
        assert(ptr != nullptr && "Allocation failed for large block size");

        pBuddy->deallocate(ptr, largeBlockSize, alignof(std::max_align_t));
    }

    // === Test 3: Exhaustion and Reallocation ===
    {
        std::vector<void*> allocations;
        size_t const       blockSize = 256;
        while (true)
        {
            void* ptr = pBuddy->allocate(blockSize, alignof(std::max_align_t));
            if (!ptr)
                break; // Stop allocating once the pool is exhausted
            allocations.push_back(ptr);
        }

        // Ensure all allocated memory is deallocated
        for (void* ptr : allocations)
            pBuddy->deallocate(ptr, blockSize, alignof(std::max_align_t));

        // Verify memory can be reallocated after full deallocation
        void* ptr = pBuddy->allocate(blockSize, alignof(std::max_align_t));
        assert(ptr != nullptr && "Reallocation after exhaustion failed");
        pBuddy->deallocate(ptr, blockSize, alignof(std::max_align_t));
    }

    // === Test 4: Basic CUDA Kernel Test ===
    {
        constexpr size_t  bufferSize = 1024; // Buffer size in bytes
        constexpr uint8_t fillValue  = 42;   // Value to fill the buffer with

        // Allocate device memory using the BuddyMemoryResource
        void* deviceBuffer = pBuddy->allocate(bufferSize, alignof(std::max_align_t));
        assert(deviceBuffer != nullptr && "Failed to allocate device buffer");

        // Launch the kernel to fill the buffer
        int threadsPerBlock = 256;
        int blocksPerGrid   = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;
        fillBufferKernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<uint8_t*>(deviceBuffer), bufferSize, fillValue);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        assert(err == ::cudaSuccess && "Kernel launch failed");

        // Synchronize to ensure kernel execution is complete
        err = cudaDeviceSynchronize();
        assert(err == ::cudaSuccess && "Device synchronization failed");

        // Allocate host memory for verification
        std::vector<uint8_t> hostBuffer(bufferSize);

        // Copy the device buffer back to the host
        err = cudaMemcpy(hostBuffer.data(), deviceBuffer, bufferSize, ::cudaMemcpyDeviceToHost);
        assert(err == ::cudaSuccess && "Failed to copy device buffer to host");

        // Verify the buffer contents
        std::string content;
        for (size_t i = 0; i < bufferSize; ++i)
        {
            content += std::to_string(hostBuffer[i]) + ", ";
            assert(hostBuffer[i] == fillValue && "Buffer verification failed");
        }
        content.resize(content.size() - 2);

        actx.log("Buffer content: \{");
        std::string_view str       = content;
        size_t           remaining = str.size();
        size_t const     maxPrint  = actx.maxLogArgBytes() >> 1;
        size_t           offset    = 0;
        while (offset < str.size())
        {
            size_t           toPrint = std::min(remaining, actx.maxLogArgBytes());
            std::string_view s       = str.substr(offset, toPrint);
            actx.log(" {}", {s});
            offset += toPrint;
            remaining -= toPrint;
        }
        actx.log("\} End Buffer content");

        // Clean up
        pBuddy->deallocate(deviceBuffer, bufferSize, alignof(std::max_align_t));
        actx.log("CUDA Kernel Test Passed {}", {"success"sv});
    }
}
