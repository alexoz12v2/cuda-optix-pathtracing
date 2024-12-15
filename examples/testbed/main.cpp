module;

#include <atomic>
#include <iostream>
#include <numbers>
#include <string_view>
#include <thread>
#include <vector>

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
    if (platform.ctx().logEnabled())
        platform.ctx().log("We are in the platform now");
}