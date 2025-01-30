
#include "cuda-tests.h"

#include <array>
#include <atomic>
#include <bit>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>
#include <cstdint>

// include cuda interface headers, not exported by C++20 modules
#define DMT_ENTRY_POINT
#include <platform/platform.h>
#include <middleware/middleware.h>
#include "platform/cudaTest.h"
#include <platform/platform-cuda-utils.h>
#include <middleware/middleware-model.h>


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
        dmt::WordParser parser;

        size_t tokenCapacity = 8;
        void*  rawMemory     = std::malloc(tokenCapacity * sizeof(CharBuffer));
        if (!rawMemory)
        {
            assert(false);
            actx.error("Couldn't allocate stack memory. Stack buffer too small or memory exhausted");
            return;
        }
        auto* arrayPtr = static_cast<CharBuffer*>(rawMemory);

        std::unique_ptr<CharBuffer[]> tokens{arrayPtr};

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
        std::free(rawMemory);
    }

    void testJob(dmt::AppContext& actx)
    {
        using namespace std::string_view_literals;
        actx.kickJobs();

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

    void textParsing(dmt::AppContext& actx)
    {
        using namespace std::string_view_literals;
        dmt::Options          opt;
        dmt::SceneDescription desc;
        auto                  filePath = "../res/scenes/disney-cloud/disney-cloud.pbrt"sv;
        dmt::SceneParser      parser{actx, &desc, filePath};

        parser.parse(actx, opt);
        std::cin.get();
    }

    void testBuddy(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemResBuddy)
    {
        testBuddyDirectly(actx, pMemResBuddy);
    }

    void testPool(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemResPool)
    {
        testMemPoolAsyncDirectly(actx, pMemResPool);
    }

    void testLightEntity()
    {
        using namespace dmt;
        AppContextJanitor j;
        j.actx.log("testLightEntity");
        std::cin.get();
        ParamMap          infiniteParams;
        ELightType        lightType = ELightType::eInfinite;
        AnimatedTransform t{Transform(), 0.f, Transform(), 1.f};
        sid_t             medium = "air"_side; // empty = void, `MakeNamedMedium`
        j.actx.log("Created shared objects");
        std::cin.get();
        infiniteParams.try_emplace("filename"_side, ParamPair{"string"_side, {"textures/spruit_sunrise_4k-clamp10.exr"}});
        infiniteParams.try_emplace("scale"_side, ParamPair{"float"_side, {"10"}});
        infiniteParams
            .try_emplace("portal"_side,
                         ParamPair{"point3"_side,
                                   {"96", "280", "-523", "96", "280", "-269", "96", "9", "-269", "96", "9", "-523"}});
        infiniteParams.try_emplace("L"_side, ParamPair{"blackbody"_side, {"6500"}});
        EColorSpaceType colorSpace = EColorSpaceType::eSRGB;
        j.actx.log("Created param map");
        std::cin.get();

        LightEntity entity{lightType, t, medium, colorSpace, infiniteParams};

        j.actx.log("Created entity");
        std::cin.get();
        std::string entityStr = "Entity (Infinite Light) {{ ";
        entityStr += *reinterpret_cast<std::string*>(&entity.spec.params.infinite.filename);
        entityStr += " portal {{ ";
        for (uint32_t i = 0; i < 4; ++i)
        {
            auto& p = entity.spec.params.infinite;
            entityStr += "( ";
            entityStr += std::to_string(p.portal[0].x);
            entityStr += ", ";
            entityStr += std::to_string(p.portal[0].y);
            entityStr += ", ";
            entityStr += std::to_string(p.portal[0].z);
            entityStr += ") ";
        }
        entityStr += "}} }";
        j.actx.log("{}", {entityStr});
    }

} // namespace

int guardedMain()
{
    dmt::AppContext actx{512, 8192, {4096, 4096, 4096, 4096}};
    dmt::ctx::init(actx);
    auto env = dmt::os::getEnv();

    //testLightEntity();
    //std::cin.get();

    actx.log("Hello darkness my old friend, {}", {sizeof(dmt::Options)});
    //dmt::model::test(actx);
    //testCTrie(actx);
    //testWordTokenization(actx);
    //testJob(actx);
    textParsing(actx);
    dmt::CUDAHelloInfo info = dmt::cudaHello(&actx.mctx());
    actx.log("CUDA Initialized");
    /*
    std::vector<float> v3;
    v3.resize(32);
    {
        std::unique_ptr<float[]> v1 = std::make_unique<float[]>(32);
        std::fill_n(v1.get(), 32, 1.f);
        std::unique_ptr<float[]> v2 = std::make_unique<float[]>(32);
        std::fill_n(v2.get(), 32, 2.f);
        dmt::kernel(v1.get(), v2.get(), 3.f, v3.data(), 32);
    }

    // create allocators
    dmt::UnifiedMemoryResource* unified = dmt::UnifiedMemoryResource::create();
    { // no allocator/alloocation should outlive unified
        actx.log("Unified Memory Resource Constructed");
        dmt::BuddyResourceSpec buddySpec{
            .pmctx        = &actx.mctx(),
            .pHostMemRes  = std::pmr::get_default_resource(),
            .maxPoolSize  = 4ULL << 20, // 1ULL << 30,
            .minBlockSize = 256,
            .minBlocks    = (2ULL << 20) / 256,
            .deviceId     = info.device,
        };
        dmt::AllocBundle buddy{unified, dmt::EMemoryResourceType::eHost, dmt::EMemoryResourceType::eHostToDevMemMap, &buddySpec};
        actx.log("Buddy Memory Resource constructed");
        dmt::MemPoolAsyncMemoryResourceSpec poolSpec{
            .pmctx            = &actx.mctx(),
            .poolSize         = 2ULL << 20, // 2MB is the minimum allocation granularity for most devices (cc 7.0)
            .releaseThreshold = std::numeric_limits<size_t>::max(),
            .pHostMemRes      = std::pmr::get_default_resource(),
            .deviceId         = info.device,
        };
        dmt::AllocBundle pool{unified, dmt::EMemoryResourceType::eAsync, dmt::EMemoryResourceType::eMemPool, &poolSpec};
        actx.log("Pool Async Memory Resource constructed");


        testBuddy(actx, buddy.pMemRes);
        testPool(actx, pool.pMemRes);

        auto* dynaArrayUnified = reinterpret_cast<dmt::DynaArray*>(
            unified->allocate(sizeof(dmt::DynaArray), alignof(dmt::DynaArray)));
        std::construct_at(dynaArrayUnified, sizeof(float), pool.pMemRes);

        testDynaArrayDirectly(actx, *dynaArrayUnified);

        std::destroy_at(dynaArrayUnified);
        unified->deallocate(dynaArrayUnified, sizeof(dmt::DynaArray), alignof(dmt::DynaArray));
        actx.log("\nPrass anything To exit");
    }

    dmt::UnifiedMemoryResource::destroy(unified);
    std::cout << "Goodbye!" << std::endl;
    */
    std::cout << "Press Any key to exit" << std::endl;
    std::cin.get();
    dmt::ctx::unregister();
    return 0;
}