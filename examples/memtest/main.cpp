
#include <cstdint>

import platform;


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
    dmt::PageAllocator      pageAllocator{ctx, testhooks};
    dmt::AllocatorHooks     defHooks;
    dmt::StackAllocator     stackAllocator{ctx, pageAllocator, defHooks};
    dmt::MultiPoolAllocator multiPoolAllocator{ctx, pageAllocator, {8192, 4096, 2048, 1024}, defHooks};

    ctx.log("Hello darkness my old friend");

    stackAllocator.cleanup(ctx, pageAllocator);
    multiPoolAllocator.cleanup(ctx, pageAllocator);
}