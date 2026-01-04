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

// TODO Use stbi to encoding in PNG
void writeOutputBuffer(float4 const* d_outputBuffer, uint32_t const width,
                       uint32_t const height, char const* name = "output.bmp",
                       bool isHost = false);
void writeOutputBufferRowMajor(float4 const* outputBuffer, uint32_t const width,
                               uint32_t const height,
                               char const* name = "output.bmp");

// --------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------
void cornellBox(bool megakernel, HostTriangleScene* h_scene,
                std::vector<Light>* h_lights,
                std::vector<Light>* h_infiniteLights,
                std::vector<BSDF>* h_bsdfs, DeviceCamera* h_camera);

#endif