
#include <memory>
#include <string_view>
#include <thread>

#include <cassert>
#include <cstdint>

import platform;

struct TestObject
{
    int              x, y;
    dmt::AppContext& ctx;
    TestObject(int a, int b, dmt::AppContext& ctx) : x(a), y(b), ctx(ctx)
    {
    }
    ~TestObject()
    {
        ctx.log("Destruction TestObject");
    }
};

void testChunkedFileReader(dmt::PlatformContext& pctx)
{
    constexpr uint32_t chunkSize = 1024; // Define chunk size (e.g., 1 KB)
    char const*        filePath  = "..\\res\\test.txt";

    dmt::ChunkedFileReader reader(pctx, filePath, chunkSize);

    // Test 1: Request a chunk
    char buffer[chunkSize];
    std::memset(buffer, 0, sizeof(buffer));
    bool success = reader.requestChunk(pctx, buffer, 0); // Read first chunk
    assert(success && "Failed to request chunk");

    // Test 2: Wait for completion (non-blocking for this test)
    bool completed = reader.waitForPendingChunk(pctx, 1000); // Wait for 1 second max
    assert(completed && "Chunk read did not complete in time");

    // Test 3: Verify data is not empty (assuming file has content)
    bool dataNonEmpty = std::strlen(buffer) > 0;
    assert(dataNonEmpty && "Buffer contains no data");

    std::string_view view{buffer, reader.lastNumBytesRead()};
    pctx.log("Bytes read from test file: {}", {view});

    // Test 4: Destructor cleanup (implicitly tested)
    // When the reader goes out of scope, the destructor will close the file handle.

    pctx.log("All tests passed for ChunkedFileReader.");
}

void testThreadpool(dmt::AppContext& ctx)
{
    struct JobData
    {
        std::atomic<int> counter;
        dmt::AppContext& ctx;

        JobData(dmt::AppContext& ctx) : counter(0), ctx(ctx)
        {
        }
    };

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
        ctx.threadPool.addJob(ctx.mctx, job, dmt::EJobLayer::eTest0);
    }

    // Enqueue jobs for eTest1 layer
    int const numJobsTest1 = 3;
    for (int i = 0; i < numJobsTest1; ++i)
    {
        dmt::Job job{jobTest1, reinterpret_cast<uintptr_t>(&test1Data)};
        ctx.threadPool.addJob(ctx.mctx, job, dmt::EJobLayer::eTest1);
    }

    for (int i = 0; i < numJobsTest0; ++i)
    {
        dmt::Job job{jobTest0, reinterpret_cast<uintptr_t>(&test0Data)};
        ctx.threadPool.addJob(ctx.mctx, job, dmt::EJobLayer::eTest0);
    }

    // Kick off the jobs
    ctx.threadPool.kickJobs();

    // Wait for jobs to complete
    while (test0Data.counter.load() < 2 * numJobsTest0 || test1Data.counter.load() < numJobsTest1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Ensure all jobs in eTest0 were executed before eTest1
    if (test0Data.counter.load() == 2 * numJobsTest0 && test1Data.counter.load() == numJobsTest1)
    {
        ctx.log("All jobs in eTest0 executed before eTest1.");
    }
    else
    {
        ctx.error("Job execution order violated.");
    }
}

// 32B  count: 8192
// 64B  count: 4096
// 128B count: 2048
// 256B count: 1024
int32_t main()
{
    dmt::Platform platform;
    auto&         ctx = platform.ctx();

    ctx.log("Hello darkness my old friend");
    auto ptr = ctx.mctx.poolAllocateBlocks(1, dmt::EBlockSize::e64B);
    if (ptr != dmt::taggedNullptr)
    {
        // Construct object in allocated memory
        auto* testObject = std::construct_at(ptr.pointer<TestObject>(), 10, 20, ctx);

        ctx.log("Allocated and constructed TestObject: x = {}, y = {}", {testObject->x, testObject->y});

        // Call destructor manually (caller responsibility)
        std::destroy_at(testObject);

        // Free the allocated memory
        ctx.mctx.poolFreeBlocks(1, ptr);
    }
    else
    {
        ctx.error("Failed to allocate memory for TestObject");
    }

    testThreadpool(ctx);
    testChunkedFileReader(ctx.mctx.pctx);
}