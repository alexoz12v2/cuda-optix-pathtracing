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
    int x, y;
    TestObject(int a, int b) : x(a), y(b) {}
    ~TestObject()
    {
        dmt::Context ctx;
        ctx.log("Destruction TestObject", {});
    }
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
    using namespace std::string_view_literals;
    constexpr uint32_t dataChunkSize = 512;
    constexpr uint32_t dataAlignment = 8;
    dmt::os::Path      filePath      = dmt::os::Path::executableDir() / "test.txt";
    uint8_t            numBuffers    = 12;
    size_t const       memSize       = dmt::os::ChunkedFileReader::computeAlignedChunkSize(dataChunkSize);
    dmt::Context       ctx;

    ctx.log("Starting ChunkedFileReader tests in PData mode", {});

    // allocate the buffers
    size_t dataSize     = memSize * numBuffers;
    size_t pointersSize = sizeof(uintptr_t) * numBuffers;
    size_t sz           = dataSize + pointersSize;

    // TODO memory reource custom
    auto*    data = reinterpret_cast<uintptr_t*>(std::pmr::get_default_resource()->allocate(sz, dataAlignment));
    auto     addr = std::bit_cast<uintptr_t>(data);
    uint64_t off  = numBuffers * sizeof(uintptr_t);
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
        std::pmr::string           filePathStr = filePath.toUnderlying();
        dmt::os::ChunkedFileReader reader{filePathStr.c_str(), dataChunkSize, numBuffers, data};
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

    ctx.log("The string read from the file is {}...", std::make_tuple(str.substr(0, 230)));
    std::pmr::get_default_resource()->deallocate(data, sz, dataAlignment);
}

static void testChunkedFileReader()
{
    constexpr uint32_t chunkSize = 512; // Define chunk size (e.g., 1 KB)
    char const*        filePath  = "..\\res\\test.txt";
    dmt::Context       ctx;

    dmt::os::ChunkedFileReader reader(filePath, chunkSize);

    // Test 1: Request a chunk
    char buffer[chunkSize];
    std::memset(buffer, 0, sizeof(buffer));
    bool success = reader.requestChunk(buffer, 0); // Read first chunk
    assert(success && "Failed to request chunk");

    // Test 2: Wait for completion (non-blocking for this test)
    bool completed = reader.waitForPendingChunk(1000); // Wait for 1 second max
    assert(completed && "Chunk read did not complete in time");

    // Test 3: Verify data is not empty (assuming file has content)
    bool dataNonEmpty = std::strlen(buffer) > 0;
    assert(dataNonEmpty && "Buffer contains no data");

    std::string_view view{buffer, std::min(reader.lastNumBytesRead(), 25u)};
    ctx.log("Bytes read from test file: {}", std::make_tuple(view));

    // Test 4: Destructor cleanup (implicitly tested)
    // When the reader goes out of scope, the destructor will close the file handle.

    ctx.log("All tests passed for ChunkedFileReader in uData mode.", {});
}

void testThreadpool()
{
    dmt::Context ctx;
    struct JobData
    {
        std::atomic<int> counter;

        JobData() : counter(0) {}
    };
    dmt::ThreadPoolV2 threadPool;

    // Create job data shared between jobs
    JobData test0Data;
    JobData test1Data;

    // Define jobs for layer eTest0
    auto jobTest0 = [](uintptr_t data) {
        dmt::Context ctx;
        JobData*     jobData = reinterpret_cast<JobData*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        ++jobData->counter;
        ctx.log("Executed job in eTest0 layer, counter: {}", std::make_tuple(jobData->counter.load()));
    };

    // Define jobs for layer eTest1
    auto jobTest1 = [](uintptr_t data) {
        dmt::Context ctx;
        JobData*     jobData = reinterpret_cast<JobData*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate work
        ++jobData->counter;
        ctx.log("Executed job in eTest1 layer, counter: {}", std::make_tuple(jobData->counter.load()));
    };

    // Enqueue jobs for eTest0 layer
    int const numJobsTest0 = 5;
    for (int i = 0; i < numJobsTest0; ++i)
    {
        dmt::Job job{jobTest0, reinterpret_cast<uintptr_t>(&test0Data)};
        threadPool.addJob(job, dmt::EJobLayer::eTest0);
    }

    // Enqueue jobs for eTest1 layer
    int const numJobsTest1 = 3;
    for (int i = 0; i < numJobsTest1; ++i)
    {
        dmt::Job job{jobTest1, reinterpret_cast<uintptr_t>(&test1Data)};
        threadPool.addJob(job, dmt::EJobLayer::eTest1);
    }

    for (int i = 0; i < numJobsTest0; ++i)
    {
        dmt::Job job{jobTest0, reinterpret_cast<uintptr_t>(&test0Data)};
        threadPool.addJob(job, dmt::EJobLayer::eTest0);
    }

    // Kick off the jobs
    threadPool.kickJobs();

    // Wait for jobs to complete
    while (test0Data.counter.load() < 2 * numJobsTest0 || test1Data.counter.load() < numJobsTest1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Ensure all jobs in eTest0 were executed before eTest1
    if (test0Data.counter.load() == 2 * numJobsTest0 && test1Data.counter.load() == numJobsTest1)
    {
        ctx.log("All jobs in eTest0 executed before eTest1. ThreadPool tests Finished", {});
    }
    else
    {
        ctx.error("Job execution order violated.", {});
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
        ctx.log("Hello darkness my old friend", {});

        testThreadpool();
        testChunkedFileReaderPData();
        testChunkedFileReader();
    }

    return 0;
}