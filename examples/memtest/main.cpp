
#include <memory>

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
        .allocHook       = [](void* data, dmt::PlatformContext& ctx, dmt::AllocationInfo const& alloc) {
			auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
            tracker.track(ctx, alloc);
        },
        .freeHook        = [](void* data, dmt::PlatformContext& ctx, dmt::AllocationInfo const& alloc) {
			auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
            tracker.untrack(ctx, alloc);
        },
        .cleanTransients = [](void* data, dmt::PlatformContext& ctx) {
			auto& tracker = *reinterpret_cast<dmt::PageAllocationsTracker*>(data);
        },
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

    stackAllocator.cleanup(ctx, pageAllocator);
    multiPoolAllocator.cleanup(ctx, pageAllocator);
}