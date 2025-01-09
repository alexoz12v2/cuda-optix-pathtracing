
#include "cuda-tests.h"

#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>
#include <cstdint>

import platform;
import middleware;

namespace {
    std::string_view input = R"=(
Integrator "volpath" "integer maxdepth" [100]
Sampler "sobol" "integer pixelsamples" [1024]
Film "rgb" 
  "integer xresolution" [1280]
  "integer yresolution" [720]
  "string filename" "disney-cloud-720p.exr"
  #"integer xresolution" [1998]
  #"integer yresolution" [1080]
  #"string filename" "disney-cloud-hd.exr"
#Reverse X axis.
Scale -1 1 1
LookAt
  #Eye
  648.064 -82.473 -63.856
  #Target
  6.021 100.043 -43.679
  #Up vector
  0.273 0.962 -0.009
#Converting fov according to the height/width ratio: 31.07 = 360/(2*PI) * 2*arctan(180/333*tan(54.43 * PI/360)).
Camera "perspective" "float fov" [31.07]
WorldBegin
#Uniform illumination from all directions.
LightSource "infinite" "rgb L" [0.03 0.07 0.23]
#Approximate the sun.
LightSource "distant"
  "point3 to" [-0.5826 -0.7660 -0.2717]
  "rgb L" [2.6 2.5 2.3]
AttributeBegin
  Translate 0 -1000 0
  Scale 2000 2000 2000
  Rotate -90 1 0 0
  Material "diffuse" "spectrum reflectance" [200 0.2 900 0.2]
  Shape "disk"  
AttributeEnd
AttributeBegin
  MakeNamedMedium "cloud" "string type" "nanovdb" 
    "string filename" "wdas_cloud_quarter.nvdb"
    "spectrum sigma_a" [200 0 900 0]
    "spectrum sigma_s" [200 1 900 1]
    "float g" [0.877]
    "float scale" [4]
  AttributeBegin
    Translate -9.984 73.008 -42.64
    Scale 206.544 140.4 254.592
    Material "interface"
    MediumInterface "cloud" ""
      Shape "sphere" "float radius" [1.44224957031]
  AttributeEnd
AttributeEnd
)=";
    void             testWordTokenization(dmt::AppContext& actx)
    {
        using CharBuffer = std::array<char, 256>;
        dmt::OneShotStackMemoryResource mem{&actx.mctx};
        dmt::WordParser                 parser;

        size_t tokenCapacity = 8;
        void*  rawMemory     = mem.allocate(tokenCapacity * sizeof(CharBuffer), alignof(CharBuffer));
        if (!rawMemory)
        {
            assert(false);
            actx.error("Couldn't allocate stack memory. Stack buffer too small or memory exhausted");
            return;
        }
        auto* arrayPtr = static_cast<CharBuffer*>(rawMemory);

        std::unique_ptr<CharBuffer[], dmt::StackArrayDeleter<CharBuffer>> tokens{arrayPtr,
                                                                                 dmt::StackArrayDeleter<CharBuffer>{}};

        size_t   chunkSize    = 512;
        size_t   offset       = 0;
        bool     stop         = false;
        uint32_t currentIndex = 0;

        while (offset < input.size())
        {
            bool             needAdvance = true;
            std::string_view chunk       = input.substr(offset, chunkSize);
            while (needAdvance)
            {
                std::string_view token = parser.nextWord(chunk);
                if (token.empty() && !parser.needsContinuation())
                { // you found only whitespaces, need to go to next chunk
                    chunk       = chunk.substr(parser.numCharReadLast());
                    needAdvance = false;
                }
                else if (!parser.needsContinuation())
                {
                    chunk = chunk.substr(parser.numCharReadLast());
                    std::memcpy(tokens[currentIndex].data(), token.data(), token.size());
                    tokens[currentIndex][token.size()] = '\0';
                    ++currentIndex;
                    if (currentIndex >= tokenCapacity)
                    { // Consume tokens
                        currentIndex = 0;
                    }
                }
                else
                {
                    needAdvance = false;
                }
            }

            offset += chunkSize;
        }
    }

    void testJob(dmt::AppContext& actx)
    {
        using namespace std::string_view_literals;
        actx.threadPool.kickJobs();

        dmt::job::ParseSceneHeaderData jobData{};
        jobData.filePath = "../res/scenes/disney-cloud/disney-cloud.pbrt"sv;
        jobData.actx     = &actx;

        dmt::Job const job{.func = dmt::job::parseSceneHeader, .data = std::bit_cast<uintptr_t>(&jobData)};

        // all preceding writes must complete before the store explicit
        std::atomic_thread_fence(std::memory_order_release);
        std::atomic_store_explicit(&jobData.done, 0, std::memory_order_relaxed);

        actx.addJob(job, dmt::EJobLayer::eTest0);

        while (std::atomic_load_explicit(&jobData.done, std::memory_order_relaxed) == 0)
        {
            std::this_thread::yield();
        }
        // all following reads must happen after the atomic retrieval
        std::atomic_thread_fence(std::memory_order_acquire);
        actx.warn("Finished job");
    }

    constexpr auto doNothing = [](dmt::MemoryContext& mctx, void* value) {};
    void           testCTrie(dmt::AppContext& actx)
    {
        dmt::CTrie ctrie{actx.mctx, dmt::AllocatorTable::fromPool(actx.mctx), sizeof(uint32_t), alignof(uint32_t)};
        uint32_t   testValue  = 43;
        uint32_t   testValue2 = 80;
        ctrie.insert(actx.mctx, 0x0000'0000, &testValue);
        ctrie.insert(actx.mctx, 0x2900'0000, &testValue2);
        ctrie.remove(actx.mctx, 0x2900'0000);
        actx.log("inserted value {}", {testValue});
        ctrie.cleanup(actx.mctx, doNothing);
    }

    void textParsing(dmt::AppContext& actx)
    {
        using namespace std::string_view_literals;
        dmt::Options opt;
        //dmt::HeaderTokenizer parser{opt};
        auto filePath = "../res/scenes/disney-cloud/disney-cloud.pbrt"sv;

        //parser.parseHeader(actx, filePath);
        dmt::TokenStream tokenStream{actx, filePath};
        for (std::string token = tokenStream.next(actx); !token.empty(); token = tokenStream.next(actx))
        {
            actx.log("sdfdsa {}", {token});
        }
        actx.log("sdfdsa");
    }

    void testBuddy(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemResBuddy)
    {
        testBuddyDirectly(actx, pMemResBuddy);
    }

    void testPool(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemResPool)
    {
        testMemPoolAsyncDirectly(actx, pMemResPool);
    }
} // namespace

int32_t main()
{
    dmt::Platform platform;
    auto&         actx = platform.ctx();
    actx.log("Hello darkness my old friend, {}", {sizeof(dmt::Options)});
    //testCTrie(actx);
    //testWordTokenization(actx);
    //testJob(actx);
    //textParsing(actx);
    dmt::CUDAHelloInfo info = dmt::cudaHello(&actx.mctx);

    dmt::EMemoryResourceType const mem     = dmt::makeMemResId(dmt::EMemoryResourceType::eDevice,
                                                           dmt::EMemoryResourceType::eCudaMalloc);
    size_t const                   memSz   = dmt::sizeForMemoryResource(mem);
    size_t const                   align   = dmt::alignForMemoryResource(mem);
    void*                          storage = alloca(memSz + align - 1);
    storage                                = dmt::alignTo(storage, align);
    dmt::BaseMemoryResource* pMemRes       = dmt::constructMemoryResourceAt(storage, mem, nullptr);
    if (void* p = pMemRes->allocateBytes(sizeof(float) * 16, alignof(float)); p)
    {
        actx.log("Allocated Device memory!");
        pMemRes->freeBytes(p, sizeof(float) * 16, alignof(float));
    }
    else
    {
        actx.error("Couldn't allocate device memory");
    }

    float*         ptr = ::new float[10];
    dmt::DynaArray arr{sizeof(float), pMemRes, dmt::noStream};
    arr.reserve(10);
    for (uint32_t i = 0; i < 10; ++i)
    {
        float f = i;
        arr.push_back(&f, true);
    }
    arr.copyToHostSync(ptr);

    std::string buf = "{ ";
    for (uint32_t i = 0; i < arr.size(); ++i)
    {
        buf += std::to_string(ptr[i]);
        buf += ", ";
    }
    buf += " }";
    actx.log("Float array: {}", {buf});

    ::delete[] ptr;

    dmt::BuddyResourceSpec buddySpec{
        .pmctx        = &actx.mctx,
        .pHostMemRes  = std::pmr::get_default_resource(),
        .maxPoolSize  = 4ULL << 20, // 1ULL << 30,
        .minBlockSize = 256,
        .minBlocks    = (2ULL << 20) / 256,
        .deviceId     = info.device,
    };
    auto const memBuddy = dmt::makeMemResId(dmt::EMemoryResourceType::eHost, dmt::EMemoryResourceType::eHostToDevMemMap);
    size_t const             memSzBuddy       = dmt::sizeForMemoryResource(memBuddy);
    size_t const             alignBuddy       = dmt::alignForMemoryResource(memBuddy);
    void*                    pMemBytes        = std::malloc(memSzBuddy + alignBuddy - 1);
    void*                    pMemBytesAligned = dmt::alignTo(pMemBytes, alignBuddy);
    dmt::BaseMemoryResource* pMemResBuddy     = dmt::constructMemoryResourceAt(pMemBytesAligned, memBuddy, &buddySpec);

    testBuddy(actx, pMemResBuddy);

    dmt::MemPoolAsyncMemoryResourceSpec poolSpec{
        .pmctx            = &actx.mctx,
        .poolSize         = 2ULL << 20, // 2MB is the minimum allocation granularity for most devices (cc 7.0)
        .releaseThreshold = std::numeric_limits<size_t>::max(),
        .pHostMemRes      = std::pmr::get_default_resource(),
        .deviceId         = info.device,
    };
    auto const   memPoolEnum  = dmt::makeMemResId(dmt::EMemoryResourceType::eAsync, dmt::EMemoryResourceType::eMemPool);
    size_t const memSzPool    = dmt::sizeForMemoryResource(memPoolEnum);
    size_t const memAlignPool = dmt::alignForMemoryResource(memPoolEnum);
    void*        pMemPoolBytes           = std::malloc(memSzPool + memAlignPool - 1);
    void*        pMemPoolBytesAligned    = dmt::alignTo(pMemPoolBytes, memAlignPool);
    dmt::BaseMemoryResource* pMemResPool = dmt::constructMemoryResourceAt(pMemPoolBytesAligned, memPoolEnum, &poolSpec);

    testPool(actx, pMemResPool);

    dmt::destroyMemoryResouceAt(pMemResPool, memPoolEnum);
    dmt::destroyMemoryResouceAt(pMemResBuddy, memBuddy);
    dmt::destroyMemoryResouceAt(pMemRes, mem);
    std::free(pMemBytes);
    std::free(pMemPoolBytes);
    actx.log("Hello darkness my old friend, {}", {sizeof(dmt::Options)});
}