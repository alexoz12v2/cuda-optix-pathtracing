#include "custuff.h"
#include "testshared/stuff.h"

#include <iostream>

#include <cstdint>

int32_t main()
{
    using namespace std::string_view_literals;
    // Hello stuff
    dmt::AppContext actx{512, 8192, {4096, 4096, 4096, 4096}};
    dmt::ctx::init(actx);
    actx.log("{} should be 3", {dmt::add(4u, 3u)});
    dmt::StringTable strTable;
    dmt::sid_t       sid = strTable.intern("fdsafsf"sv);
    actx.log("{}", {strTable.lookup(sid)});

    dmt::CUDAHelloInfo info = dmt::cudaHello(nullptr);

    // create allocators
    dmt::UnifiedMemoryResource unified;
    dmt::BuddyResourceSpec     buddySpec{
            .pmctx        = nullptr,
            .pHostMemRes  = std::pmr::get_default_resource(),
            .maxPoolSize  = 4ULL << 20, // 1ULL << 30,
            .minBlockSize = 256,
            .minBlocks    = (2ULL << 20) / 256,
            .deviceId     = info.device,
    };
    dmt::AllocBundle buddy{&unified, dmt::EMemoryResourceType::eHost, dmt::EMemoryResourceType::eHostToDevMemMap, &buddySpec};
    dmt::MemPoolAsyncMemoryResourceSpec poolSpec{
        .pmctx            = nullptr,
        .poolSize         = 2ULL << 20, // 2MB is the minimum allocation granularity for most devices (cc 7.0)
        .releaseThreshold = std::numeric_limits<size_t>::max(),
        .pHostMemRes      = std::pmr::get_default_resource(),
        .deviceId         = info.device,
    };
    dmt::AllocBundle pool{&unified, dmt::EMemoryResourceType::eAsync, dmt::EMemoryResourceType::eMemPool, &poolSpec};

    // create CUDA vector
    {
        auto* dynaArrayUnified = reinterpret_cast<dmt::DynaArray*>(
            unified.allocate(sizeof(dmt::DynaArray), alignof(dmt::DynaArray)));
        std::construct_at(dynaArrayUnified, sizeof(float), pool.pMemRes);

        dynaArrayUnified->reserve(32);
        std::unique_ptr<float[]> a = std::make_unique<float[]>(dynaArrayUnified->capacity());
        fillVector(*dynaArrayUnified, 5.f, 7.f, a.get());

        actx.log("Vec: { ");
        dynaArrayUnified->copyToHostSync(a.get());
        std::string str = "\n";
        for (size_t i = 0; i < dynaArrayUnified->size(); ++i)
        {
            str += std::to_string(a[i]);
            str += " ";
            if (i % 8 == 0)
                str += "\n      ";
        }
        actx.log("{}", {str});

        std::destroy_at(dynaArrayUnified);
        unified.deallocate(dynaArrayUnified, sizeof(dmt::DynaArray), alignof(dmt::DynaArray));
    }

    std::cout << "Press Any key to exit" << std::endl;
    std::cin.get();
    std::cout << "Goodbye!" << std::endl;
    dmt::ctx::unregister();
}