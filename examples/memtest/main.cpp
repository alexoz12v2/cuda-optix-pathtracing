#define DMT_ENTRY_POINT
#include <platform/platform.h>

#include <bit>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include <cassert>
#include <cstdint>
#include <cstdio>

static void worker(void* unused)
{
    dmt::Context ctx;
    for (int i = 0; i < 5; ++i)
    {
        ctx.log("Thread running... {}", std::make_tuple(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


int guardedMain()
{
    using namespace std::string_view_literals;
    dmt::Ctx::init();
    struct Janitor
    {
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
        dmt::Context ctx;
        ctx.log("Starting memory management tests...", {});

        static constexpr size_t testSize      = dmt::toUnderlying(dmt::EPageSize::e2MB);
        static constexpr size_t largePageSize = dmt::toUnderlying(dmt::EPageSize::e1GB);

        // Test: Reserve Virtual Address Space
        void* reservedAddress = dmt::os::reserveVirtualAddressSpace(testSize);
        if (reservedAddress)
        {
            ctx.log("Successfully reserved virtual address space", {});
        }
        else
        {
            ctx.error("Failed to reserve virtual address space", {});
        }

        // Test: Commit Physical Memory
        bool commitSuccess = dmt::os::commitPhysicalMemory(reservedAddress, testSize);
        if (commitSuccess)
        {
            ctx.log("Successfully committed physical memory", {});
        }
        else
        {
            ctx.error("Failed to commit physical memory", {});
        }

        // Test: Decommit Physical Memory
        dmt::os::decommitPhysicalMemory(reservedAddress, testSize);
        ctx.log("Decommitted physical memory", {});

        // Test: Free Virtual Address Space
        bool freeSuccess = dmt::os::freeVirtualAddressSpace(reservedAddress, testSize);
        if (freeSuccess)
        {
            ctx.log("Successfully freed virtual address space", {});
        }
        else
        {
            ctx.error("Failed to free virtual address space", {});
        }

        // Test: Allocate Locked Large Pages (2MB)
        void* largePageMemory2MB = dmt::os::allocateLockedLargePages(testSize, dmt::EPageSize::e2MB, false);
        if (largePageMemory2MB)
        {
            ctx.log("Successfully allocated locked large pages (2MB)", {});
            dmt::os::deallocateLockedLargePages(largePageMemory2MB, testSize, dmt::EPageSize::e2MB);
            ctx.log("Successfully deallocated locked large pages (2MB)", {});
        }
        else
        {
            ctx.error("Failed to allocate locked large pages (2MB)", {});
        }

        // Test: Allocate Locked Large Pages (1GB)
        void* largePageMemory1GB = dmt::os::allocateLockedLargePages(largePageSize, dmt::EPageSize::e1GB, false);
        if (largePageMemory1GB)
        {
            ctx.log("Successfully allocated locked large pages (1GB)", {});
            dmt::os::deallocateLockedLargePages(largePageMemory1GB, largePageSize, dmt::EPageSize::e1GB);
            ctx.log("Successfully deallocated locked large pages (1GB)", {});
        }
        else
        {
            ctx.error("Failed to allocate locked large pages (1GB)", {});
        }

        ctx.log("Memory management tests completed.", {});

        dmt::os::Thread t{worker};
        ctx.log("Running thread {}", std::make_tuple(t.id()));
        t.start();

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        ctx.log("Joining Thread {}", std::make_tuple(t.id()));
        t.join();
        ctx.log("Thread Joined", {});

        ctx.log("Starting Pool Allocator memory tests", {});

        // Create a SyncPoolAllocator instance
        dmt::SyncPoolAllocator allocator(dmt::EMemoryTag::eUnknown, 1024 * 1024, 64, dmt::EBlockSize::e256B); // 1MB reserved, 256B block size, 64 initial blocks

        // Check if allocator is valid
        if (!allocator.isValid())
        {
            ctx.error("Allocator is not valid!", {});
            return -1; // Early exit if invalid
        }

        // Allocate memory: Try to allocate 512 bytes
        void* ptr = allocator.allocate(512);
        if (ptr != nullptr)
        {
            ctx.log("Successfully allocated 512 bytes.", {});
        }
        else
        {
            ctx.error("Allocation failed!", {});
            return -1;
        }

        // Allocate more memory: Try to allocate another 128 bytes
        void* ptr2 = allocator.allocate(128);
        if (ptr2 != nullptr)
        {
            ctx.log("Successfully allocated another 128 bytes.", {});
        }
        else
        {
            ctx.error("Second allocation failed!", {});
            return -1;
        }

        // Deallocate memory: Free the first pointer (512 bytes)
        allocator.deallocate(ptr, 512);
        ctx.log("Successfully deallocated 512 bytes.", {});

        // Deallocate second memory block (128 bytes)
        allocator.deallocate(ptr2, 128);
        ctx.log("Successfully deallocated 128 bytes.", {});

        // Attempting allocation after deallocation
        void* ptr3 = allocator.allocate(256);
        if (ptr3 != nullptr)
        {
            ctx.log("Successfully allocated 256 bytes after deallocation.", {});
        }
        else
        {
            ctx.error("Allocation after deallocation failed!", {});
            return -1;
        }

        // Check number of blocks in the allocator
        uint32_t numBlocks = allocator.numBlocks();
        ctx.log("Number of blocks in the allocator: {}", std::make_tuple(numBlocks));
    }

    return 0;
}
