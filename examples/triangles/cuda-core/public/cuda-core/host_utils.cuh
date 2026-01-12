#ifndef DMT_CUDA_CORE_HOST_UTILS_CUH
#define DMT_CUDA_CORE_HOST_UTILS_CUH

#include "cuda-core/common_math.cuh"
#include "cuda-core/bsdf.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/host_scene.cuh"
#include "cuda-core/rng.cuh"

#include <cassert>
#include <cstdint>
#include <memory>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <queue>
#include <ranges>

// TODO cornell box builder

// --------------------------------------------------------------------------
// Upload Functions
// --------------------------------------------------------------------------
DeviceHaltonOwen* copyHaltonOwenToDeviceAlloc(uint32_t blocks,
                                              uint32_t threads);

// TODO remove when switching to shapes/BVH
TriangleSoup triSoupFromTriangles(const HostTriangleScene& hostScene,
                                  uint32_t const bsdfCount,
                                  size_t maxTrianglesPerChunk = 1'000'000);
BSDF* deviceBSDF(std::vector<BSDF> const& h_bsdfs);
void deviceLights(std::vector<Light> const& h_lights,
                  std::vector<Light> const& h_infiniteLights, Light** d_lights,
                  Light** d_infiniteLights);
DeviceCamera* deviceCamera(DeviceCamera const& h_camera);

// --------------------------------------------------------------------------
// General Utils
// --------------------------------------------------------------------------
float4* deviceOutputBuffer(uint32_t const width, uint32_t const height);

std::filesystem::path getExecutableDirectory();

template <bool PreferHighTPB = false>
__host__ void optimalOccupancyFromBlock(void* krnl, uint32_t smemBytes,
                                        bool residentOnly, uint32_t& blocks,
                                        uint32_t& threads) {
  cudaDeviceProp deviceProp{};
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  struct OccupancyMeasure {
    int blocks, threads;
    bool operator<(OccupancyMeasure const& other) const {
      if constexpr (PreferHighTPB)
        return threads < other.threads;
      else
        return blocks < other.blocks;
    }
  };
  std::priority_queue<OccupancyMeasure> occupancies;
  // instead of using cudaDeviceProp.maxThreadsPerBlock, cap it to 512 as
  // it would hurt occupancy due to register pressure
  auto multiples = std::views::iota(1, 16) |
                   std::views::transform([](int i) { return i * 32; }) |
                   std::views::reverse;  // 512, 480, 448 ... 32
  for (uint32_t t : multiples) {
    OccupancyMeasure current{.blocks = 0, .threads = (int)t};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &current.blocks, krnl, current.threads, smemBytes));
    // round down to multiple of number of SMs
    current.blocks = (current.blocks / deviceProp.multiProcessorCount) *
                     deviceProp.multiProcessorCount;
    // account for current only if we don't care about residency of threads
    // or blocks per multiprocessor fits inside SM
    if (!residentOnly ||
        deviceProp.maxThreadsPerMultiProcessor >=
            current.blocks / deviceProp.multiProcessorCount * current.threads) {
      occupancies.emplace(current);
    }
  }
  blocks = occupancies.top().blocks;
  threads = occupancies.top().threads;
}

void writeOutputBuffer(float4 const* d_outputBuffer, uint32_t const width,
                       uint32_t const height, char const* name = "output.bmp",
                       bool isHost = false);
void writeOutputBufferRowMajor(float4 const* outputBuffer, uint32_t const width,
                               uint32_t const height,
                               char const* name = "output.bmp");

// --------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------
void cornellBox(HostTriangleScene* h_scene, std::vector<Light>* h_lights,
                std::vector<Light>* h_infiniteLights,
                std::vector<BSDF>* h_bsdfs, DeviceCamera* h_camera);

#endif