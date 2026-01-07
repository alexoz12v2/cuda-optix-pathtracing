#include "wave-kernels.cuh"
#include "cuda-core/types.cuh"
#include "cuda-core/bsdf.cuh"
#include "cuda-core/rng.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/extra_math.cuh"
#include "cuda-core/host_scene.cuh"
#include "cuda-core/host_utils.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/morton.cuh"
#include "cuda-core/shapes.cuh"
#include "cuda-core/kernels.cuh"
#include "cuda-core/queue.cuh"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>
#include <device_types.h>

#include <cassert>

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

namespace cg = cooperative_groups;

namespace {

#if 0
__host__ void optimalBlocksAndThreads(uint32_t& blocks, uint32_t& threads,
                                      uint32_t& sharedBytes) {
  // ensure that blocks can be all resident in the GPU at once
  // - either use 1 block per SM (*)
  // - or use the Occupancy API to figure out max blocks
  cudaDeviceProp deviceProp{};
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  // To be kept in sync with kernel consumption
  // RG: 4 bytes (warpRemaining)
  // sinks: 256 bytes (16 x 8 bytes (addresses) = 1 address per 2 banks
  sharedBytes = 12;
#  if 1
  int desiredThreads = 32;
#  else
  int desiredThreads = 512;
#  endif
  int availableBlocks = 0;
  while (availableBlocks < blocks && desiredThreads != 0) {
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &availableBlocks, wavefrontKernel, desiredThreads, sharedBytes));
    if (availableBlocks < blocks) {
      desiredThreads >>= 1;
    }
  }
  if (desiredThreads < WARP_SIZE || availableBlocks < blocks) {
    std::cerr << "\033[31mCooperativeLaunch blocks couldn't be satisfied\033[0m"
              << std::endl;
    exit(1);
  }

  assert(desiredThreads % WARP_SIZE == 0);
  threads = desiredThreads;
}
#endif

struct CallbackData {
  float4* hostImage;
  uint32_t width;
  uint32_t height;
  uint32_t sample;
};

void CUDART_CB streamCallbackWriteBuffer(void* userData) {
  auto* data = static_cast<CallbackData*>(userData);
  std::cout << "Storing Result (" << data->sample << ")" << std::endl;
  std::string const name =
      "wave-stream-output-" + std::to_string(data->sample) + ".bmp";
  writeOutputBufferRowMajor(data->hostImage, data->width, data->height,
                            name.c_str());
}

void wavefrontMain() {
  CUDA_CHECK(cudaInitDevice(0, 0, 0));

  // set cache configuration
  CUDA_CHECK(cudaFuncSetCacheConfig(raygenKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(closesthitKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(anyhitKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(missKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(shadeKernel, cudaFuncCachePreferL1));

  uint32_t threads = 1;
  uint32_t blocks = 1;
  uint32_t sharedBytes = 12;
  // optimalBlocksAndThreads(blocks, threads, sharedBytes); // TODO
  std::cout << "Computed Optimal Occupancy for wavefront kernels: " << blocks
            << " blocks. " << threads << " threads. " << sharedBytes
            << " SMEM Bytes" << std::endl;

  // init scene
  std::cout << "Allocating host and device resources" << std::endl;
  HostTriangleScene h_scene;
  std::vector<Light> h_lights;
  std::vector<Light> h_infiniteLights;
  std::vector<BSDF> h_bsdfs;
  DeviceCamera h_camera;
  cornellBox(true, &h_scene, &h_lights, &h_infiniteLights, &h_bsdfs, &h_camera);

  allocateDeviceConstantMemory(h_camera);

  auto* kinput = new WavefrontStreamInput(threads, blocks, h_scene, h_lights,
                                          h_infiniteLights, h_bsdfs, h_camera);
  kinput->sampleOffset = 0;

  // TODO double buffering
  float4* hostImage = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostImage,
                            sizeof(float4) * h_camera.width * h_camera.height));
  auto const callbackData = std::make_unique<CallbackData>();
  callbackData->hostImage = hostImage;
  callbackData->width = h_camera.width;
  callbackData->height = h_camera.height;

  // TODO double buffering
  cudaStream_t st_main;
  CUDA_CHECK(cudaStreamCreate(&st_main));

  static int constexpr TOTAL_SAMPLES = 2048;
  for (int sOffset = 0; sOffset < TOTAL_SAMPLES; sOffset += h_camera.spp) {
    kinput->sampleOffset = sOffset;
    callbackData->sample = kinput->sampleOffset;
    std::cout << "Launching kernel (" << kinput->sampleOffset << ")"
              << std::endl;

    raygenKernel<<<blocks, threads, sharedBytes, st_main>>>(
        kinput->closesthitQueue, kinput->pathStateSlots, kinput->d_haltonOwen,
        kinput->d_cam, kinput->sampleOffset);
    CUDA_CHECK(cudaGetLastError());

    closesthitKernel<<<blocks, threads, sharedBytes, st_main>>>(
        kinput->closesthitQueue, kinput->missQueue, kinput->anyhitQueue,
        kinput->shadeQueue, kinput->d_haltonOwen, kinput->d_triSoup);
    CUDA_CHECK(cudaGetLastError());

    anyhitKernel<<<blocks, threads, sharedBytes, st_main>>>(
        kinput->anyhitQueue, kinput->d_haltonOwen, kinput->d_lights,
        kinput->lightCount, kinput->d_bsdfs, kinput->d_triSoup);
    CUDA_CHECK(cudaGetLastError());

    shadeKernel<<<blocks, threads, sharedBytes, st_main>>>(
        kinput->shadeQueue, kinput->closesthitQueue, kinput->pathStateSlots,
        kinput->d_outBuffer, kinput->d_haltonOwen, kinput->d_bsdfs);
    CUDA_CHECK(cudaGetLastError());

    closesthitKernel<<<blocks, threads, sharedBytes, st_main>>>(
        kinput->closesthitQueue, kinput->missQueue, kinput->anyhitQueue,
        kinput->shadeQueue, kinput->d_haltonOwen, kinput->d_triSoup);
    CUDA_CHECK(cudaGetLastError());

    missKernel<<<blocks, threads, sharedBytes, st_main>>>(
        kinput->missQueue, kinput->pathStateSlots, kinput->d_outBuffer,
        kinput->d_haltonOwen, kinput->infiniteLights,
        kinput->infiniteLightCount);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemcpyAsync(hostImage, kinput->d_outBuffer,
                        h_camera.width * h_camera.height * sizeof(float4),
                        cudaMemcpyDeviceToHost, st_main));

    CUDA_CHECK(cudaStreamSynchronize(st_main));
    streamCallbackWriteBuffer(callbackData.get());
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "Cleanup..." << std::endl;

  CUDA_CHECK(cudaStreamDestroy(st_main));

  CUDA_CHECK(cudaFreeHost(hostImage));
  delete kinput;
  freeDeviceConstantMemory();
}

}  // namespace

// UNICODE and _UNICODE always defined
#ifdef _WIN32
int wmain() {
#else
int main() {
#endif
#ifdef DMT_OS_WINDOWS
  SetConsoleOutputCP(CP_UTF8);
  for (DWORD conoutHandleId : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
    HANDLE const hConsole = GetStdHandle(conoutHandleId);
    DWORD mode = 0;
    if (GetConsoleMode(hConsole, &mode)) {
      mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
      SetConsoleMode(hConsole, mode);
    }
  }
#endif
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  CUDA_CHECK(cudaSetDevice(0));
  wavefrontMain();
}
