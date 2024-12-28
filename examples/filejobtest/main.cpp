
#include <atomic>
#include <bit>
#include <string_view>
#include <thread>

#include <cstdint>

import platform;
import middleware;

int32_t main()
{
    using namespace std::string_view_literals;
    dmt::Platform platform;
    auto&         actx = platform.ctx();
    actx.log("Hello darkness my old friend, {}", {sizeof(dmt::Options)});

    dmt::job::ParseSceneHeaderData jobData{};
    jobData.filePath = "../res/scenes/disney-cloud/disney-cloud.pbrt"sv;
    jobData.actx     = &actx;

    dmt::Job const job{.func = dmt::job::parseSceneHeader, .data = std::bit_cast<uintptr_t>(&jobData)};

    // all preceding writes must complete before the store explicit
    std::atomic_thread_fence(std::memory_order_release);
    std::atomic_store_explicit(&jobData.done, 0, std::memory_order_relaxed);

    actx.addJob(job, dmt::EJobLayer::eTest0);
    actx.threadPool.kickJobs();

    while (std::atomic_load_explicit(&jobData.done, std::memory_order_relaxed) == 0)
    {
        std::this_thread::yield();
    }
    // all following reads must happen after the atomic retrieval
    std::atomic_thread_fence(std::memory_order_acquire);
    actx.warn("Finished job");
}