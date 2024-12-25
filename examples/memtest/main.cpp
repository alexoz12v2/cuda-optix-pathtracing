
#include <memory>
#include <thread>

#include <cstdint>

import platform;

struct TestObject
{
    int                   x, y;
    dmt::PlatformContext& ctx;
    TestObject(int a, int b, dmt::PlatformContext& ctx) : x(a), y(b), ctx(ctx)
    {
    }
    ~TestObject()
    {
        ctx.log("Destruction TestObject");
    }
};

void testThreadpool(dmt::PlatformContext&    ctx,
                    dmt::PageAllocator&      pageAllocator,
                    dmt::StackAllocator&     stackAllocator,
                    dmt::MultiPoolAllocator& multiPoolAllocator)
{
    struct JobData
    {
        std::atomic<int>      counter;
        dmt::PlatformContext& ctx;

        JobData(dmt::PlatformContext& ctx) : counter(0), ctx(ctx)
        {
        }
    };

    // Instantiate the thread pool
    dmt::ThreadPoolV2 threadPool(ctx, pageAllocator, multiPoolAllocator, stackAllocator);

    // Create job data shared between jobs
    JobData test0Data{ctx};
    JobData test1Data{ctx};

    // Define jobs for layer eTest0
    auto jobTest0 = [](uintptr_t data)
    {
        JobData* jobData = reinterpret_cast<JobData*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        ++jobData->counter;
        jobData->ctx.log("Executed job in eTest0 layer, counter: {}", {jobData->counter.load()});
    };

    // Define jobs for layer eTest1
    auto jobTest1 = [](uintptr_t data)
    {
        JobData* jobData = reinterpret_cast<JobData*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate work
        ++jobData->counter;
        jobData->ctx.log("Executed job in eTest1 layer, counter: {}", {jobData->counter.load()});
    };

    // Enqueue jobs for eTest0 layer
    int const numJobsTest0 = 5;
    for (int i = 0; i < numJobsTest0; ++i)
    {
        dmt::Job job{jobTest0, reinterpret_cast<uintptr_t>(&test0Data)};
        threadPool.addJob(ctx, pageAllocator, multiPoolAllocator, job, dmt::EJobLayer::eTest0);
    }

    // Enqueue jobs for eTest1 layer
    int const numJobsTest1 = 3;
    for (int i = 0; i < numJobsTest1; ++i)
    {
        dmt::Job job{jobTest1, reinterpret_cast<uintptr_t>(&test1Data)};
        threadPool.addJob(ctx, pageAllocator, multiPoolAllocator, job, dmt::EJobLayer::eTest1);
    }

    // Kick off the jobs
    threadPool.kickJobs();

    // Wait for jobs to complete
    while (test0Data.counter.load() < numJobsTest0 || test1Data.counter.load() < numJobsTest1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Ensure all jobs in eTest0 were executed before eTest1
    if (test0Data.counter.load() == numJobsTest0 && test1Data.counter.load() == numJobsTest1)
    {
        ctx.log("All jobs in eTest0 executed before eTest1.");
    }
    else
    {
        ctx.error("Job execution order violated.");
    }

    // Cleanup the thread pool
    threadPool.cleanup(ctx, pageAllocator, multiPoolAllocator);
}

// 32B  count: 8192
// 64B  count: 4096
// 128B count: 2048
// 256B count: 1024
int32_t main()
{
    dmt::Platform               platform;
    auto&                       ctx = platform.ctx();
    dmt::PageAllocationsTracker tracker{ctx, dmt::toUnderlying(dmt::EPageSize::e1GB), false};
    dmt::PageAllocatorHooks     testhooks{
            .allocHook =
            [](void* data, dmt::PlatformContext& ctx, dmt::PageAllocation const& alloc) { //
                auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
                tracker.track(ctx, alloc);
            },
            .freeHook =
            [](void* data, dmt::PlatformContext& ctx, dmt::PageAllocation const& alloc) { //
                auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
                tracker.untrack(ctx, alloc);
            },
            .data = &tracker,
    };
    dmt::AllocatorHooks defHooks{
        .allocHook =
            [](void* data, dmt::PlatformContext& ctx, dmt::AllocationInfo const& alloc)
        {
            auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
            tracker.track(ctx, alloc);
        },
        .freeHook =
            [](void* data, dmt::PlatformContext& ctx, dmt::AllocationInfo const& alloc)
        {
            auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
            tracker.untrack(ctx, alloc);
        },
        .cleanTransients = [](void* data, dmt::PlatformContext& ctx)
        { auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data); },
        .data = &tracker,
    };

    dmt::PageAllocator      pageAllocator{ctx, testhooks};
    dmt::StackAllocator     stackAllocator{ctx, pageAllocator, defHooks};
    dmt::MultiPoolAllocator multiPoolAllocator{ctx, pageAllocator, {8192, 4096, 2048, 1024}, defHooks};

    ctx.log("Hello darkness my old friend");
    auto ptr = multiPoolAllocator.allocateBlocks(ctx, pageAllocator, 1, dmt::EBlockSize::e64B);
    if (ptr != dmt::taggedNullptr)
    {
        // Construct object in allocated memory
        auto* testObject = std::construct_at(ptr.pointer<TestObject>(), 10, 20, ctx);

        ctx.log("Allocated and constructed TestObject: x = {}, y = {}", {testObject->x, testObject->y});

        // Call destructor manually (caller responsibility)
        std::destroy_at(testObject);

        // Free the allocated memory
        multiPoolAllocator.freeBlocks(ctx, pageAllocator, 1, ptr);
    }
    else
    {
        ctx.error("Failed to allocate memory for TestObject");
    }

    testThreadpool(ctx, pageAllocator, stackAllocator, multiPoolAllocator);

    stackAllocator.cleanup(ctx, pageAllocator);
    multiPoolAllocator.cleanup(ctx, pageAllocator);
}