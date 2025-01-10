module;

#include <atomic>
#include <iostream>
#include <memory>
#include <numbers>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>
#include <cstring>

module platform;

namespace // all functions declared in an anonymous namespace (from the global namespace) are static by default
{
    void printSome(dmt::ConsoleLogger& logger)
    {
        using namespace std::string_view_literals;
        logger.log("Hello World from logger");
        logger.warn("Hello Warn from logger");
        logger.error("Hello error from logger");
        logger.log("Hello World from logger");
        logger.log("Hello {} from logger", {"world"sv});
    }

    void testLoggingInMultithreadedEnvironment(dmt::ConsoleLogger& logger,
                                               int32_t numThreads = static_cast<int32_t>(std::thread::hardware_concurrency()))
    {
        // Vector to hold all the threads
        std::vector<std::jthread> threads;

        // Atomic counter to ensure all threads are finished before printing the final result
        std::atomic<int32_t> completedThreads{0};

        // Lambda function for thread execution
        auto logTask = [&logger, &completedThreads](int32_t id) {
            using namespace std::string_view_literals;

            // Each thread logs different messages
            logger.log("Thread ID {}: Hello World from logger", {id});
            logger.warn("Thread ID {}: Hello Warn from logger", {id});
            logger.error("Thread ID {}: Hello error from logger", {id});
            logger.log("Thread ID {}: Hello World from logger", {id});
            logger.log("Thread ID {}: Hello {} from logger", {id, "multithreaded"sv});

            // Increment the atomic counter to signal completion
            completedThreads.fetch_add(1, std::memory_order_relaxed);
        };

        // Spawn multiple threads
        for (int32_t i = 0; i < numThreads; ++i)
        {
            threads.emplace_back(logTask, i);
        }

        // Wait for all threads to complete
        for (auto& t : threads)
        {
            t.join();
        }

        // Ensure all threads completed
        logger.log("All threads completed logging. Total threads: {}", {completedThreads.load()});
    }

    void testStackAllocator(dmt::LoggingContext& ctx, dmt::PageAllocator& pageAllocator, dmt::StackAllocator& stackAllocator)
    {
        // Allocate memory with different sizes and alignments
        void* ptr1 = stackAllocator.allocate(ctx, pageAllocator, 128, 16);
        assert(ptr1 != nullptr && reinterpret_cast<uintptr_t>(ptr1) % 16 == 0);
        ctx.log("Allocated 128 bytes aligned to 16 at {}", {ptr1});

        void* ptr2 = stackAllocator.allocate(ctx, pageAllocator, 256, 32);
        assert(ptr2 != nullptr && reinterpret_cast<uintptr_t>(ptr2) % 32 == 0);
        ctx.log("Allocated 256 bytes aligned to 32 at {}", {ptr2});

        // Reset the allocator and allocate again
        stackAllocator.reset(ctx, pageAllocator);
        ctx.log("Allocator reset.");

        void* ptr3 = stackAllocator.allocate(ctx, pageAllocator, 512, 64);
        assert(ptr3 != nullptr && reinterpret_cast<uintptr_t>(ptr3) % 64 == 0);
        ctx.log("Allocated 512 bytes aligned to 64 at {}", {ptr3});

        // Attempt to allocate more than buffer size (should fail gracefully)
        void* ptr4 = stackAllocator.allocate(ctx, pageAllocator, 3 * 1024 * 1024, 128); // 3 MB
        assert(ptr4 == nullptr);
        ctx.log("Failed to allocate 3 MB as expected.");

        // Cleanup resources
        stackAllocator.cleanup(ctx, pageAllocator);
        ctx.log("Allocator cleaned up successfully.");
    }

    void testAllocations(dmt::LoggingContext&         ctx,
                         dmt::PageAllocator&          pageAllocator,
                         dmt::PageAllocationsTracker& tracker,
                         uint32_t&                    counter)
    {
        size_t   numBytes       = dmt::toUnderlying(dmt::EPageSize::e1GB);
        uint32_t numAllocations = 0;
        ctx.log("Attempting request to allocate {} Bytes, {} GB", {numBytes, numBytes >> 30u});

        auto pageSize    = pageAllocator.allocatePagesForBytesQuery(ctx, numBytes, numAllocations);
        auto allocations = std::make_unique<dmt::PageAllocation[]>(numAllocations);
        auto allocInfo = pageAllocator.allocatePagesForBytes(ctx, numBytes, allocations.get(), numAllocations, pageSize);
        ctx.log("Actually allocated {} Bytes, {} MB, {} GB",
                {allocInfo.numBytes, allocInfo.numBytes >> 20u, allocInfo.numBytes >> 30u});
        counter = 0;

        for (auto const& node : tracker.pageAllocations())
        {
            ctx.log("Tracker Data: Allocated page at {}, frame number {} of size {}",
                    {node.data.alloc.address, node.data.alloc.pageNum, (void*)dmt::toUnderlying(node.data.alloc.pageSize)});
        }

        for (uint32_t i = 0; i != allocInfo.numPages; ++i)
        {
            dmt::PageAllocation& ref = allocations[i];
            if (ref.address != nullptr)
            {
                pageAllocator.deallocPage(ctx, ref);
            }
        }
    }

} // namespace

int main()
{
    using namespace std::string_view_literals;
    dmt::CircularOStringStream oss;
    char const*                formatStr = "this is a \\{} {} string. Pi: {}, 4 pi: {}, 1000 == {}, thuthy: {}\n";
    float                      pi        = std::numbers::pi_v<float>;
    bool                       b         = true;
    int                        thou      = 1000;
    std::string_view           arg{"format"};
    oss.logInitList(formatStr, {arg, pi, dmt::StrBuf(pi, "%.5f"), thou, b});
    std::cout << oss.str() << std::endl;
    dmt::ConsoleLogger logger = dmt::ConsoleLogger::create();
    printSome(logger);
    testLoggingInMultithreadedEnvironment(logger);
    logger.trace("I shall not be seen");

    dmt::Platform platform;
    auto&         ctx = platform.ctx().mctx.pctx;
    if (platform.ctx().logEnabled())
        platform.ctx().log("We are in the platform now");

    uint32_t                counter = 0;
    dmt::PageAllocatorHooks hooks{
        .allocHook =
            [](void* data, dmt::LoggingContext& ctx, dmt::PageAllocation const& alloc) { //
        uint32_t& counter = *reinterpret_cast<uint32_t*>(data);
        if (counter++ % 50 == 0)
            ctx.log("Inside allocation hook!");
    },
        .freeHook =
            [](void* data, dmt::LoggingContext& ctx, dmt::PageAllocation const& alloc) { //
        uint32_t& counter = *reinterpret_cast<uint32_t*>(data);
        if (counter++ % 50 == 0)
            ctx.log("inside deallocation Hook!");
    },
        .data = &counter,
    };
    dmt::PageAllocationsTracker tracker{ctx, dmt::toUnderlying(dmt::EPageSize::e1GB), false};
    dmt::PageAllocatorHooks     testhooks{
            .allocHook =
            [](void* data, dmt::LoggingContext& ctx, dmt::PageAllocation const& alloc) { //
        auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
        tracker.track(ctx, alloc);
    },
            .freeHook =
            [](void* data, dmt::LoggingContext& ctx, dmt::PageAllocation const& alloc) { //
        auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
        tracker.untrack(ctx, alloc);
    },
            .data = &tracker,
    };
    dmt::PageAllocator pageAllocator{ctx, testhooks};
    auto               pageAlloc = pageAllocator.allocatePage(platform.ctx());
    if (pageAlloc.address)
    {
        platform.ctx().log("Allocated page at {}, frame number {} of size {}",
                           {pageAlloc.address, pageAlloc.pageNum, (void*)dmt::toUnderlying(pageAlloc.pageSize)});
    }
    else
    {
        platform.ctx().error("Couldn't allocate memory");
    }
    pageAllocator.deallocPage(platform.ctx(), pageAlloc);

    platform.ctx().log("Completed");

    std::string_view str = "thishtisdfasdf"sv;
    dmt::sid_t       sid = dmt::operator""_sid(str.data(), str.size());
    platform.ctx().log("{}", {dmt::lookupInternedStr(sid)});

    //testAllocations(ctx, pageAllocator, tracker, counter);

    dmt::AllocatorHooks defHooks;
    dmt::StackAllocator stackAllocator{ctx, pageAllocator, defHooks};
    testStackAllocator(ctx, pageAllocator, stackAllocator);
    stackAllocator.cleanup(ctx, pageAllocator);
}