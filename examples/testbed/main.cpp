module;

#include <atomic>
#include <iostream>
#include <numbers>
#include <string_view>
#include <thread>
#include <vector>
#include <memory>

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
    auto logTask = [&logger, &completedThreads](int32_t id)
    {
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
    auto&         ctx = platform.ctx();
    if (platform.ctx().logEnabled())
        platform.ctx().log("We are in the platform now");

    dmt::PageAllocator pageAllocator{platform.ctx()};
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
    pageAllocator.deallocatePage(platform.ctx(), pageAlloc);

    platform.ctx().log("Completed");

    std::string_view str = "thishtisdfasdf"sv;
    dmt::sid_t  sid = dmt::operator""_sid(str.data(), str.size());
    platform.ctx().log("{}", {dmt::lookupInternedStr(sid)});

    size_t numBytes = dmt::toUnderlying(dmt::EPageSize::e1GB);
    uint32_t numAllocations = 0;
    ctx.log("Attempting request to allocate {} Bytes, {} GB", {numBytes, numBytes >> 30u });

    auto pageSize = pageAllocator.allocatePagesForBytesQuery(platform.ctx(), numBytes, numAllocations, false);
    auto allocations = std::make_unique<dmt::PageAllocation[]>(numAllocations);
    uint32_t num = pageAllocator.allocatePagesForBytes(platform.ctx(), numBytes, allocations.get(), numAllocations, pageSize);
    size_t allocatedBytes = 0;
    for (uint32_t i = 0; i != num; ++i)
    {
        dmt::PageAllocation& ref = allocations[i];
        allocatedBytes += dmt::toUnderlying(ref.pageSize);
    }
    ctx.log("Actually allocated {} Bytes, {} MB, {} GB", { allocatedBytes, allocatedBytes >> 20u, allocatedBytes >> 30u });

    for (uint32_t i = 0; i != num; ++i)
    {
        dmt::PageAllocation& ref = allocations[i];
        if (ref.address != nullptr)
        {
            pageAllocator.deallocatePage(platform.ctx(), ref);
        }
    }
}