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

#  if DMT_ENABLE_ASSERTS
  assert(desiredThreads % WARP_SIZE == 0);
#  endif
  threads = desiredThreads;
}
#endif

struct CallbackData {
#if DMT_ENABLE_MSE
  HostPinnedOutputBuffer h_out;
#else
  float4* hostImage;
#endif
  uint32_t width;
  uint32_t height;
  uint32_t sample;
};

void CUDART_CB streamCallbackWriteBuffer(void* userData) {
  auto* data = static_cast<CallbackData*>(userData);
  std::cout << "Storing Result (" << data->sample << ")" << std::endl;
#if DMT_ENABLE_MSE
  std::string const name = "wave-stream-output-" + std::to_string(data->sample);
#  if 0
  writeMeanAndMSERowMajorCompHost(data->h_out.meanPtr, data->h_out.m2Ptr,
                                  data->width, data->height, name,
                                  data->sample);
#  else
  writeMeanAndMSERowMajor(data->h_out.meanPtr, data->h_out.m2Ptr, data->width,
                          data->height, name);
#  endif
#else
  std::string const name =
      "wave-stream-output-" + std::to_string(data->sample) + ".png";
  writeOutputBufferRowMajor(data->hostImage, data->width, data->height,
                            name.c_str());
#endif
}

void wavefrontMain() {
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  // input parsing
  Config config = parseArguments(getPlatformArgs(), true);
  if (auto const err = config.validate(); !err.empty()) {
    std::cerr << err << std::endl;
    Config::printHelp();
    exit(1);
  }
  config.print();

  // set cache configuration
  CUDA_CHECK(cudaFuncSetCacheConfig(raygenKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(closesthitKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(anyhitKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(missKernel, cudaFuncCachePreferL1));
  CUDA_CHECK(cudaFuncSetCacheConfig(shadeKernel, cudaFuncCachePreferL1));

  uint32_t threads[5]{512};
  uint32_t blocks[5]{100};
  uint32_t const sharedBytes = 0;

  // minimum TPB tweakable
  optimalOccupancyFromBlock<false, 1, 128>((void*)closesthitKernel, sharedBytes,
                                           true, blocks[0], threads[0]);
  optimalOccupancyFromBlock<false, 1, 64>((void*)anyhitKernel, sharedBytes,
                                          true, blocks[1], threads[1]);
  optimalOccupancyFromBlock<false, 1, 64>((void*)shadeKernel, sharedBytes, true,
                                          blocks[2], threads[2]);
  optimalOccupancyFromBlock<false, 1, 64>((void*)missKernel, sharedBytes, true,
                                          blocks[3], threads[3]);
  optimalOccupancyFromBlock<false, 1, 256>((void*)checkDoneDepth, sharedBytes,
                                           true, blocks[4], threads[4]);

  std::cout << "Computed Optimal Occupancy for wavefront kernels: ";
  {
    std::vector<std::string> names{"closesthitKernel", "anyhitKernel",
                                   "shadeKernel", "missKernel",
                                   "checkDoneDepth"};
    uint32_t count = 0;
    for (auto const& sk : names) {
      std::cout << " - " << sk << ": blocks " << blocks[count]
                << " threads: " << threads[count] << '\n';
      ++count;
    }
    std::cout << std::flush;
  }

  // init scene
  std::cout << "Allocating host and device resources" << std::endl;
  HostTriangleScene h_scene;
  std::vector<Light> h_lights;
  std::vector<Light> h_infiniteLights;
  std::vector<BSDF> h_bsdfs;
  DeviceCamera h_camera;
  cornellBox(&h_scene, &h_lights, &h_infiniteLights, &h_bsdfs, &h_camera);
  // now from config
  h_camera.width = config.width;
  h_camera.height = config.height;
  h_camera.spp = config.kspp;
  int const TOTAL_SAMPLES = config.spp;
  bool const isVerbose = config.isLogVerbose();

  static int constexpr TILES_PER_WARP = 4;
  static int constexpr TILE_X = 8 * TILES_PER_WARP;
  static int constexpr TILE_Y = 4 * TILES_PER_WARP;

  // compute the number of tiles each loop iteration will run on
  int const numTilesX = ceilDiv(h_camera.width, TILE_X);
  int const numTilesY = ceilDiv(h_camera.height, TILE_Y);
  int const tileDimY = TILE_Y;  // TODO tunable with cmdline params
  int const tileDimX = TILE_X;

  allocateDeviceConstantMemory(h_camera, tileDimX, tileDimY);

  // TODO estimate queue size from tile size, spp, max depth
  auto* kinput = new WavefrontStreamInput(1, WARP_SIZE, h_scene, h_lights,
                                          h_infiniteLights, h_bsdfs, h_camera);
  kinput->sampleOffset = 0;

  // TODO double buffering
  size_t const outputBytes = h_camera.width * h_camera.height * sizeof(float4);
#if DMT_ENABLE_MSE
  DeviceOutputBuffer d_out{};
  d_out.allocate(h_camera.width, h_camera.height);
#else
  float4* hostImage = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostImage,
                            sizeof(float4) * h_camera.width * h_camera.height));
#endif

  auto const callbackData = std::make_unique<CallbackData>();
#if DMT_ENABLE_MSE
  callbackData->h_out.allocate(h_camera.width, h_camera.height);
#else
  callbackData->hostImage = hostImage;
#endif
  callbackData->width = h_camera.width;
  callbackData->height = h_camera.height;

  int* h_done = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_done, sizeof(int), cudaHostAllocMapped));
  int* d_done = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&d_done, h_done, 0));

  // TODO double buffering
  cudaStream_t st_main;
  CUDA_CHECK(cudaStreamCreate(&st_main));

  // timing
  AvgAndTotalTimer timer;
  std::cout << "Running CUDA Kernel" << std::endl;

#define KERNEL_DEBUG 0

  // TODO if occupancy allows, process more tiles of the image?
  for (int sOffset = 0; sOffset < TOTAL_SAMPLES; sOffset += h_camera.spp) {
    kinput->sampleOffset = sOffset;
    callbackData->sample = kinput->sampleOffset + h_camera.spp;
    if (isVerbose) {
      std::cout << "Launching kernel (" << kinput->sampleOffset << ")"
                << std::endl;
    }
    for (int tileY = 0; tileY < numTilesY; tileY++) {
      for (int tileX = 0; tileX < numTilesX; tileX++) {
        raygenKernel<<<1, WARP_SIZE, sharedBytes, st_main>>>(
            kinput->closesthitQueue, kinput->pathStateSlots,
            kinput->d_haltonOwen, kinput->d_cam, tileX, tileY, tileDimX,
            tileDimY, kinput->sampleOffset);
        CUDA_CHECK(cudaGetLastError());

#if KERNEL_DEBUG
        CUDA_CHECK(cudaStreamSynchronize(st_main));
        std::cout << "[raygenKernel] " << "s: " << callbackData->sample
                  << " tileX: " << tileX << " tileY: " << tileY << std::endl;
#endif

        *h_done = false;
        // TODO: instead of a single zero-copy boolean, estimate number of
        // blocks
        while (!*h_done) {
          closesthitKernel<<<blocks[0], threads[0], sharedBytes, st_main>>>(
              kinput->closesthitQueue, kinput->missQueue, kinput->anyhitQueue,
              kinput->shadeQueue, kinput->d_triSoup);
          CUDA_CHECK(cudaGetLastError());

#if KERNEL_DEBUG
          CUDA_CHECK(cudaStreamSynchronize(st_main));
          std::cout << "[closesthitKernel] " << "s: " << callbackData->sample
                    << " tileX: " << tileX << " tileY: " << tileY << std::endl;
#endif

          anyhitKernel<<<blocks[1], threads[1], sharedBytes, st_main>>>(
              kinput->anyhitQueue, kinput->d_lights, kinput->lightCount,
              kinput->d_bsdfs, kinput->d_triSoup);
          CUDA_CHECK(cudaGetLastError());

#if KERNEL_DEBUG
          CUDA_CHECK(cudaStreamSynchronize(st_main));
          std::cout << "[anyhitKernel] " << "s: " << callbackData->sample
                    << " tileX: " << tileX << " tileY: " << tileY << std::endl;
#endif

          shadeKernel<<<blocks[2], threads[2], sharedBytes, st_main>>>(
              kinput->shadeQueue, kinput->closesthitQueue,
              kinput->pathStateSlots,
#if DMT_ENABLE_MSE
              d_out,
#else
              kinput->d_outBuffer,
#endif
              kinput->d_bsdfs);
          CUDA_CHECK(cudaGetLastError());

#if KERNEL_DEBUG
          CUDA_CHECK(cudaStreamSynchronize(st_main));
          std::cout << "[shadeKernel] " << "s: " << callbackData->sample
                    << " tileX: " << tileX << " tileY: " << tileY << std::endl;
#endif

          missKernel<<<blocks[3], threads[3], sharedBytes, st_main>>>(
              kinput->missQueue, kinput->pathStateSlots,
#if DMT_ENABLE_MSE
              d_out,
#else
              kinput->d_outBuffer,
#endif
              kinput->infiniteLights, kinput->infiniteLightCount);
          CUDA_CHECK(cudaGetLastError());

#if KERNEL_DEBUG
          CUDA_CHECK(cudaStreamSynchronize(st_main));
          std::cout << "[missKernel] " << "s: " << callbackData->sample
                    << " tileX: " << tileX << " tileY: " << tileY << std::endl;
#endif

          // TODO With more than 100 blocks, it freezes
          // checkDoneDepth<<<blocks[4], threads[4], sharedBytes, st_main>>>(
          checkDoneDepth<<<100, WARP_SIZE, 12, st_main>>>(
              kinput->pathStateSlots, kinput->closesthitQueue,
              kinput->missQueue, kinput->anyhitQueue, kinput->shadeQueue,
              d_done);
          CUDA_CHECK(cudaGetLastError());

          kinput->swapBuffersAllQueues(st_main);
          CUDA_CHECK(cudaStreamSynchronize(st_main));

#if KERNEL_DEBUG
          CUDA_CHECK(cudaStreamSynchronize(st_main));
          std::cout << "[checkDoneDepth] " << "s: " << callbackData->sample
                    << " tileX: " << tileX << " tileY: " << tileY << std::endl;
#endif
        }
      }
    }

    if (config.savePartial) {
#if DMT_ENABLE_MSE
      CUDA_CHECK(cudaMemcpyAsync(callbackData->h_out.m2Ptr, d_out.m2Ptr,
                                 outputBytes, cudaMemcpyDeviceToHost, st_main));
      CUDA_CHECK(cudaMemcpyAsync(callbackData->h_out.meanPtr, d_out.meanPtr,
                                 outputBytes, cudaMemcpyDeviceToHost, st_main));
#else
      CUDA_CHECK(cudaMemcpyAsync(hostImage, kinput->d_outBuffer, outputBytes,
                                 cudaMemcpyDeviceToHost, st_main));
#endif
    }

    CUDA_CHECK(cudaStreamSynchronize(st_main));
    timer.tick();

    if (config.savePartial) {
      streamCallbackWriteBuffer(callbackData.get());
      timer.reset();
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  if (!config.savePartial) {
#if DMT_ENABLE_MSE
    CUDA_CHECK(cudaMemcpyAsync(callbackData->h_out.m2Ptr, d_out.m2Ptr,
                               outputBytes, cudaMemcpyDeviceToHost, st_main));
    CUDA_CHECK(cudaMemcpyAsync(callbackData->h_out.meanPtr, d_out.meanPtr,
                               outputBytes, cudaMemcpyDeviceToHost, st_main));
#else
    CUDA_CHECK(cudaMemcpyAsync(hostImage, kinput->d_outBuffer, outputBytes,
                               cudaMemcpyDeviceToHost, st_main));
#endif
    CUDA_CHECK(cudaStreamSynchronize(st_main));
    timer.tick();
    streamCallbackWriteBuffer(callbackData.get());
  }

  std::cout << "Done! Total Execution Time(excl write file): "
            << timer.elapsedMillis()
            << " ms | Average Execution per Kernel launch (" << h_camera.spp
            << " spp): " << timer.avgMillis() << " ms" << std::endl;

  std::cout << "Cleanup..." << std::endl;

  CUDA_CHECK(cudaStreamDestroy(st_main));

#if DMT_ENABLE_MSE
#else
  CUDA_CHECK(cudaFreeHost(hostImage));
#endif
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
