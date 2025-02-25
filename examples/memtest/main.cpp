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


struct TestObject
{
    int              x, y;
    dmt::AppContext& ctx;
    TestObject(int a, int b, dmt::AppContext& ctx) : x(a), y(b), ctx(ctx) {}
    ~TestObject() { ctx.log("Destruction TestObject"); }
};

static dmt::StrBuf formatAlloc(dmt::AllocationInfo const& alloc, char const* sidStr)
{
    static thread_local char storage[256]{};

    // Format the information into the storage buffer
    int len = std::snprintf(storage,
                            sizeof(storage),
                            "( Address: %p, AllocTime: %llu ms, FreeTime: %llu ms, Size: %zu bytes, "
                            "SID: %s, Alignment: %u, Transient: %u, Tag: %s )",
                            alloc.address,
                            alloc.allocTime,
                            alloc.freeTime,
                            alloc.size,
                            sidStr,
                            alloc.alignment,
                            alloc.transient,
                            dmt::memoryTagStr(alloc.tag));

    // Ensure the string is null-terminated and does not exceed the buffer
    if (len < 0 || static_cast<size_t>(len) >= sizeof(storage))
    {
        // Handle formatting errors or truncation
        storage[sizeof(storage) - 1] = '\0';
    }

    return {storage, len};
}

static void testChunkedFileReaderPData()
{
    dmt::AppContextJanitor j;
    using namespace std::string_view_literals;
    constexpr uint32_t dataChunkSize = 512;
    constexpr uint32_t dataAlignment = 8;
    char const*        filePath      = "..\\res\\test.txt";
    uint8_t            numBuffers    = 12;
    size_t const       memSize       = dmt::os::ChunkedFileReader::computeAlignedChunkSize(dataChunkSize);

    // allocate the buffers
    size_t dataSize     = memSize * numBuffers;
    size_t pointersSize = sizeof(uintptr_t) * numBuffers;
    size_t sz           = dataSize + pointersSize;

    dmt::sid_t sid = j.actx.intern("Test Read Chunked File Buffer From Stack"sv);

    auto* data = reinterpret_cast<uintptr_t*>(j.actx.stackAllocate(sz, dataAlignment, dmt::EMemoryTag::eUnknown, sid));
    auto  addr = std::bit_cast<uintptr_t>(data);
    uint64_t off = numBuffers * sizeof(uintptr_t);
    assert(data);

    // setup the array of pointers to `memSize` sized buffers
    for (uint32_t i = 0; i < numBuffers; ++i)
    {
        data[i] = dmt::alignToAddr(addr + off, dataAlignment);
        off += memSize;
    }

    std::string str{};
    // reader needs to go out of scope before freeing memory
    {
        dmt::os::ChunkedFileReader reader{j.actx.mctx().pctx, filePath, dataChunkSize, numBuffers, data};
        str.resize(dataChunkSize * reader.numChunks());
        for (dmt::ChunkInfo chunkinfo : reader.range(0, reader.numChunks()))
        {
            assert(chunkinfo.numBytesRead <= dataChunkSize);
            for (uint32_t i = 0; i < chunkinfo.numBytesRead; ++i)
            {
                str[chunkinfo.chunkNum * dataChunkSize + i] = reinterpret_cast<char*>(chunkinfo.buffer)[i];
            }
            reader.markFree(chunkinfo);
        }
    }

    j.actx.log("The string read from the file is {}...", {str.substr(0, 230)});

    // print and test memory allocation tracking
    for (auto const& alloc : j.actx.tracker().allocations())
    {
        j.actx.log("Allocation Info: {}", {formatAlloc(alloc.data.alloc, j.actx.lookup(alloc.data.alloc.sid).data())});
    }

    j.actx.stackReset();
}

static void testChunkedFileReader()
{
    dmt::AppContextJanitor j;
    constexpr uint32_t     chunkSize = 512; // Define chunk size (e.g., 1 KB)
    char const*            filePath  = "..\\res\\test.txt";

    dmt::os::ChunkedFileReader reader(j.actx.mctx().pctx, filePath, chunkSize);

    // Test 1: Request a chunk
    char buffer[chunkSize];
    std::memset(buffer, 0, sizeof(buffer));
    bool success = reader.requestChunk(j.actx.mctx().pctx, buffer, 0); // Read first chunk
    assert(success && "Failed to request chunk");

    // Test 2: Wait for completion (non-blocking for this test)
    bool completed = reader.waitForPendingChunk(j.actx.mctx().pctx, 1000); // Wait for 1 second max
    assert(completed && "Chunk read did not complete in time");

    // Test 3: Verify data is not empty (assuming file has content)
    bool dataNonEmpty = std::strlen(buffer) > 0;
    assert(dataNonEmpty && "Buffer contains no data");

    std::string_view view{buffer, std::min(reader.lastNumBytesRead(), 25u)};
    j.actx.log("Bytes read from test file: {}", {view});

    // Test 4: Destructor cleanup (implicitly tested)
    // When the reader goes out of scope, the destructor will close the file handle.

    j.actx.log("All tests passed for ChunkedFileReader in uData mode.");
}

void testThreadpool()
{
    dmt::AppContextJanitor j;
    struct JobData
    {
        std::atomic<int> counter;
        dmt::AppContext& ctx;

        JobData(dmt::AppContext& ctx) : counter(0), ctx(ctx) {}
    };

    // Create job data shared between jobs
    JobData test0Data{j.actx};
    JobData test1Data{j.actx};

    // Define jobs for layer eTest0
    auto jobTest0 = [](uintptr_t data) {
        JobData* jobData = reinterpret_cast<JobData*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        ++jobData->counter;
        jobData->ctx.log("Executed job in eTest0 layer, counter: {}", {jobData->counter.load()});
    };

    // Define jobs for layer eTest1
    auto jobTest1 = [](uintptr_t data) {
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
        j.actx.addJob(job, dmt::EJobLayer::eTest0);
    }

    // Enqueue jobs for eTest1 layer
    int const numJobsTest1 = 3;
    for (int i = 0; i < numJobsTest1; ++i)
    {
        dmt::Job job{jobTest1, reinterpret_cast<uintptr_t>(&test1Data)};
        j.actx.addJob(job, dmt::EJobLayer::eTest1);
    }

    for (int i = 0; i < numJobsTest0; ++i)
    {
        dmt::Job job{jobTest0, reinterpret_cast<uintptr_t>(&test0Data)};
        j.actx.addJob(job, dmt::EJobLayer::eTest0);
    }

    // Kick off the jobs
    j.actx.kickJobs();

    // Wait for jobs to complete
    while (test0Data.counter.load() < 2 * numJobsTest0 || test1Data.counter.load() < numJobsTest1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Ensure all jobs in eTest0 were executed before eTest1
    if (test0Data.counter.load() == 2 * numJobsTest0 && test1Data.counter.load() == numJobsTest1)
    {
        j.actx.log("All jobs in eTest0 executed before eTest1.");
    }
    else
    {
        j.actx.error("Job execution order violated.");
    }
}

// 32B  count: 8192
// 64B  count: 4096
// 128B count: 2048
// 256B count: 1024
int guardedMain()
{
    using namespace std::string_view_literals;
    dmt::AppContext actx{512, 4096, {4096, 4096, 4096, 4096}};
    dmt::ctx::init(actx);

    actx.log("Hello darkness my old friend");
    dmt::sid_t sid = actx.intern("Test Pool Allocation block");

    auto ptr = actx.poolAllocateBlocks(1, dmt::EBlockSize::e64B, dmt::EMemoryTag::eUnknown, sid);
    if (ptr != dmt::taggedNullptr)
    {
        // Construct object in allocated memory
        auto* testObject = std::construct_at(ptr.pointer<TestObject>(), 10, 20, actx);

        actx.log("Allocated and constructed TestObject: x = {}, y = {}", {testObject->x, testObject->y});

        // Call destructor manually (caller responsibility)
        std::destroy_at(testObject);

        // Free the allocated memory
        actx.poolFreeBlocks(1, ptr);
    }
    else
    {
        actx.error("Failed to allocate memory for TestObject");
    }

    testThreadpool();
    testChunkedFileReaderPData();
    testChunkedFileReader();

    // for each sliding window
    for (auto const& allocRange : actx.tracker().slidingWindow())
    {
        for (auto const& alloc : allocRange)
        {
            actx.log("Allocation Info: {}", {formatAlloc(alloc, actx.lookup(alloc.sid).data())});
        }
    }

    dmt::ctx::unregister();
    return 0;
}