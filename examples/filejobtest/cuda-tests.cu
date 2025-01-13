#define DMT_INTERFACE_AS_HEADER
#include "cuda-tests.h"
#include "dmtmacros.h"
#include "platform/platform-cuda-utils.cuh"
#include <platform/platform-cuda-utils.h>
#include "platform/platform.h"

#include <cuda_device_runtime_api.h>
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
    dmt::BuddyMemoryResource* pBuddy = reinterpret_cast<dmt::BuddyMemoryResource*>(pMemRes);
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

    // === Test 5: Multithreaded Allocation and Deallocation ===
    {
        constexpr size_t blockSize            = 512;
        constexpr int    numThreads           = 8;
        constexpr int    allocationsPerThread = 100;

        std::vector<std::thread> threads;
        std::vector<void*>       allocations[numThreads];

        // Allocate blocks in multiple threads
        for (int t = 0; t < numThreads; ++t)
        {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < allocationsPerThread; ++i)
                {
                    void* ptr = pBuddy->allocate(blockSize, alignof(std::max_align_t));
                    assert(ptr != nullptr && "Allocation failed in multithreaded test");
                    allocations[t].push_back(ptr);
                }
            });
        }

        for (auto& thread : threads)
            thread.join();

        threads.clear();

        // Deallocate blocks in multiple threads
        for (int t = 0; t < numThreads; ++t)
        {
            threads.emplace_back([&, t]() {
                for (void* ptr : allocations[t])
                    pBuddy->deallocate(ptr, blockSize, alignof(std::max_align_t));
            });
        }

        for (auto& thread : threads)
            thread.join();
    }

    // === Test 6: Copy Semantics ===
    {
        dmt::BuddyMemoryResource copy = *pBuddy; // Test copy constructor
        void*                    ptr  = copy.allocate(512, alignof(std::max_align_t));
        assert(ptr != nullptr && "Copy constructor test failed");
        copy.deallocate(ptr, 512, alignof(std::max_align_t));

        dmt::BuddyMemoryResource move = std::move(copy); // Test move constructor
        ptr                           = move.allocate(512, alignof(std::max_align_t));
        assert(ptr != nullptr && "Move constructor test failed");
        move.deallocate(ptr, 512, alignof(std::max_align_t));

        *pBuddy = move; // Test copy assignment
        ptr     = pBuddy->allocate(512, alignof(std::max_align_t));
        assert(ptr != nullptr && "Copy assignment test failed");
        pBuddy->deallocate(ptr, 512, alignof(std::max_align_t));
    }
}

void testMemPoolAsyncDirectly(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemRes)
{
    using namespace std::string_view_literals;
    auto* pPool = reinterpret_cast<dmt::MemPoolAsyncMemoryResource*>(pMemRes);
    assert(pPool && "incorrect memory resource type");
    assert(cudaGetLastError() == ::cudaSuccess);
    actx.log("Starting tests for MemPoolAsyncMemoryResource.");

    // Utility lambdas
    auto logBufferContents = [&](uint8_t const* buffer, size_t size, std::string_view context) {
        std::string content;
        for (size_t i = 0; i < size; ++i)
        {
            content += std::to_string(buffer[i]) + ", ";
            assert(buffer[i] == buffer[0] && "Buffer content mismatch detected");
        }
        content.resize(content.size() - 2);

        actx.log("Buffer content ({}): {", {context});
        std::string_view str       = content;
        size_t           remaining = str.size();
        while (remaining > 0)
        {
            size_t           chunkSize = std::min(remaining, actx.maxLogArgBytes());
            std::string_view view      = str.substr(str.size() - remaining, chunkSize);
            actx.log(" {}", {view});
            remaining -= chunkSize;
        }
        actx.log("} End Buffer content");
    };

    auto executeKernelAndCopyBack =
        [&](void* deviceBuffer, size_t bufferSize, uint8_t fillValue, std::vector<uint8_t>& hostBuffer, cudaStream_t stream) {
        fillBufferKernel<<<8, 256, 0, stream>>>(static_cast<uint8_t*>(deviceBuffer), bufferSize, fillValue);
        cudaStreamSynchronize(stream);
        assert(cudaGetLastError() == ::cudaSuccess && "Kernel execution failed");

        cudaMemcpyAsync(hostBuffer.data(), deviceBuffer, bufferSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        assert(cudaGetLastError() == ::cudaSuccess && "Memcpy failed");
    };

    // === Test 1: Mixed Allocation/Deallocation ===
    {
        actx.log("=== Test 1: Mixed Allocation/Deallocation ===");

        constexpr size_t blockSize = 256; // Size for testing
        void*            ptr1      = pPool->allocate(blockSize, alignof(std::max_align_t));
        actx.log("Allocated first block of size {}", {blockSize});

        void* ptr2 = pPool->allocate(blockSize * 2, alignof(std::max_align_t));
        actx.log("Allocated second block of size {}", {blockSize * 2});

        assert(ptr1 != ptr2 && "Allocation failed or returned duplicate pointer");

        pPool->deallocate(ptr1, blockSize, alignof(std::max_align_t));
        actx.log("Deallocated first block.");

        pPool->deallocate(ptr2, blockSize * 2, alignof(std::max_align_t));
        actx.log("Deallocated second block.");
    }

    // === Test 2: High Stress Allocation ===
    {
        actx.log("=== Test 2: High Stress Allocation ===");

        size_t poolSize = pPool->poolSize();
        actx.log("Pool size is {}", {poolSize});

        void* largeAlloc = pPool->allocate(poolSize, alignof(std::max_align_t));
        actx.log("Allocated entire pool size.");

        pPool->deallocate(largeAlloc, poolSize, alignof(std::max_align_t));
        actx.log("Deallocated entire pool size.");

        // Allocate in smaller chunks
        constexpr size_t   chunkSize = 256;
        std::vector<void*> allocations;
        size_t             allocated = 0;

        while (allocated + chunkSize <= poolSize)
        {
            void* ptr = pPool->allocate(chunkSize, alignof(std::max_align_t));
            allocations.push_back(ptr);
            allocated += chunkSize;
        }

        actx.log("Allocated {} chunks of size {}", {allocations.size(), chunkSize});

        // Deallocate chunks
        for (void* ptr : allocations)
        {
            pPool->deallocate(ptr, chunkSize, alignof(std::max_align_t));
        }
        actx.log("Deallocated all chunks.");
    }

    // === Test 3: Multithreaded Allocation ===
    {
        actx.log("=== Test 3: Multithreaded Allocation ===");

        constexpr size_t         threadCount = 8;
        constexpr size_t         allocSize   = 1024;
        std::vector<std::thread> threads;

        for (size_t i = 0; i < threadCount; ++i)
        {
            threads.emplace_back([i, &pPool, &actx]() {
                void* ptr = pPool->allocate(allocSize, alignof(std::max_align_t));
                actx.log("Thread {} allocated memory of size {}", {i, allocSize});
                pPool->deallocate(ptr, allocSize, alignof(std::max_align_t));
                actx.log("Thread {} deallocated memory.", {i});
            });
        }

        for (auto& t : threads)
            t.join();

        actx.log("All threads completed memory operations.");
    }

    // === Test 4: Kernel Execution ===
    {
        actx.log("=== Test 4: Kernel Execution ===");

        constexpr size_t  bufferSize = 1024;
        constexpr uint8_t testValue  = 42;

        void* dBuffer = pPool->allocate(bufferSize, alignof(uint8_t));
        actx.log("Allocated device buffer of size {}", {bufferSize});

        fillBufferKernel<<<8, 256>>>(static_cast<uint8_t*>(dBuffer), bufferSize, testValue);
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == ::cudaSuccess && "Kernel execution failed");
        actx.log("Kernel executed successfully.");

        std::vector<uint8_t> hBuffer(bufferSize);
        cudaMemcpy(hBuffer.data(), dBuffer, bufferSize, cudaMemcpyDeviceToHost);
        actx.log("Copied buffer back to host memory.");

        // Log and validate buffer contents
        std::string content;
        for (size_t i = 0; i < bufferSize; ++i)
        {
            content += std::to_string(hBuffer[i]) + ", ";
            assert(hBuffer[i] == testValue && "Kernel did not write expected values");
        }
        content.resize(content.size() - 2);

        actx.log("Buffer content: {");
        std::string_view str       = content;
        size_t           remaining = str.size();
        while (remaining > 0)
        {
            size_t           chunkSize = std::min(remaining, actx.maxLogArgBytes());
            std::string_view view      = str.substr(str.size() - remaining, chunkSize);
            actx.log(" {}", {view});
            remaining -= chunkSize;
        }
        actx.log("} End Buffer content");

        pPool->deallocate(dBuffer, bufferSize, alignof(uint8_t));
        actx.log("Deallocated device buffer.");
    }

    // === Test 5: Copy Control ===
    {
        actx.log("=== Test 5: Copy Control ===");

        dmt::MemPoolAsyncMemoryResource copiedPool = *pPool; // Copy constructor
        actx.log("Copied pool with size {}", {copiedPool.poolSize()});

        dmt::MemPoolAsyncMemoryResource movedPool = std::move(*pPool); // Move constructor
        actx.log("Moved pool with size {}", {movedPool.poolSize()});

        copiedPool = movedPool; // Copy assignment
        actx.log("Copied pool via assignment with size {}", {copiedPool.poolSize()});

        movedPool = std::move(copiedPool); // Move assignment
        actx.log("Moved pool via assignment with size {}", {movedPool.poolSize()});

        *pPool = std::move(movedPool);
    }

    // === Test 6: Async Allocation/Deallocation ===
    {
        actx.log("=== Test 6: Async Allocation/Deallocation ===");

        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        constexpr size_t  bufferSize = 1024;
        constexpr uint8_t fillValue1 = 42;
        constexpr uint8_t fillValue2 = 84;

        std::vector<uint8_t> hostBuffer1(bufferSize), hostBuffer2(bufferSize);

        actx.log("Allocating buffers asynchronously on two streams.", {});

        // Allocate buffers on separate streams
        void* dBuffer1 = pPool->allocate_async(bufferSize, alignof(uint8_t), stream1);
        void* dBuffer2 = pPool->allocate_async(bufferSize, alignof(uint8_t), stream2);

        actx.log("Allocated buffers of size {} on streams.", {bufferSize});

        // Execute kernels on each stream
        executeKernelAndCopyBack(dBuffer1, bufferSize, fillValue1, hostBuffer1, stream1);
        executeKernelAndCopyBack(dBuffer2, bufferSize, fillValue2, hostBuffer2, stream2);

        // Log results
        logBufferContents(hostBuffer1.data(), bufferSize, "Stream1");
        logBufferContents(hostBuffer2.data(), bufferSize, "Stream2");

        // Deallocate asynchronously
        pPool->deallocate_async(dBuffer1, bufferSize, alignof(uint8_t), stream1);
        pPool->deallocate_async(dBuffer2, bufferSize, alignof(uint8_t), stream2);

        actx.log("Deallocated buffers asynchronously.", {});

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);

        actx.log("Destroyed CUDA streams.", {});
    }

    actx.log("Completed all tests for MemPoolAsyncMemoryResource.");
}

__global__ void dynaArraytestKernel(dmt::DynaArray& dynaArray)
{ // the only error we are not covering is a device mismatch, ie ``
    int32_t device;
    if (cudaGetDevice(&device) != ::cudaSuccess)
    { // what?
        return;
    }

    if (dmt::categoryOf(dynaArray.resource()) == dmt::EMemoryResourceType::eDevice)
    { // __device__ code is allowed to make the array grow
    }
    else if (!isDeviceAllocator(dynaArray.resource(), device))
    { // __device__ code cannot access the array elements, as they are in a __host__ only region of memory
    }
    else
    { // __device__ code cannot grow the DynaArray past its current capacity
    }
}

void testDynaArrayDirectly(dmt::AppContext& actx, dmt::DynaArray& dynaArray)
{
    actx.log("----------------- Beginning Tests For DynaArray ------------------------------");
    if (dynaArray.size() != 0)
    {
        actx.error("Container should start in a clean state");
        assert(false);
        return;
    }
    if (cudaGetLastError() != ::cudaSuccess)
    {
        actx.error("CUDA runtime should start in a clean error state");
        assert(false);
        return;
    }

    actx.log("----------------- Completed Tests For DynaArray ------------------------------");
}
