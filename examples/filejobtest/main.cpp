
#include <atomic>
#include <bit>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cstdint>

import platform;
import middleware;

namespace {
    void testWordTokenization(dmt::AppContext& actx)
    {
        std::string_view input = R"(
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
        )";

        dmt::WordParser          parser;
        std::vector<std::string> tokens;
        tokens.reserve(128);

        size_t chunkSize = 512;
        size_t offset    = 0;
        bool   stop      = false;

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
                    tokens.emplace_back(token);
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
} // namespace

int32_t main()
{
    dmt::Platform platform;
    auto&         actx = platform.ctx();
    actx.log("Hello darkness my old friend, {}", {sizeof(dmt::Options)});
    testWordTokenization(actx);
    //testJob(actx);
}