#include "megakernel.cuh"

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

#include <cuda_runtime_api.h>

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

#include <cassert>
#include <cstdint>
#include <memory>
#include <fstream>
#include <filesystem>
#include <string>

// - rework triangle logic such that we switch to indexed/instanced triangles
// - add lights, hence shadow ray production and casting (without lights, we
// cannot reason about bounces)
// - work on a separate executable on BSDF function and chi2 test on these (CPU
// and GPU)

// Spaces
//   Raster space
//     Pixel coordinates (x, y)
//     Origin: top-left (0,0)
//     +X → right, +Y → down
//   Camera space
//     +X → right, +Y → up, +Z → forward
//     Origin: camera pinhole
//     Units: meters
//   World/render space
//     +X → right, +Y → forward, +Z → up
//     Origin: arbitrary(preferably camera), worldFromCamera handles camera
//       orientation

// Stream Aware operations
// - for a truly asynchronous, stream-aware cudaMemcpyAsync()
//   from host to device, the source memory must be page-locked (pinned).
// - Stack memory is normally pageable, so you must register it with
//   cudaHostRegister() if you want the copy to be asynchronous.
// - You do not need to manually find page boundaries—the CUDA runtime does
//   that for you—but if you want to know how it works or align explicitly,
//   it’s OS-specific and straightforward  cudaHostRegister()
// - if possible, don't waste time pinning a whole page for a single variable.

namespace {}  // namespace

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
namespace {

void megakernelMain() {
  CUDA_CHECK(
      cudaFuncSetCacheConfig(pathTraceMegakernel, cudaFuncCachePreferL1));
  static uint32_t constexpr OPTIMAL_SMEM_BYTES = 2048;
#if DMT_ENABLE_MSE
#else
#endif
  uint32_t threads = -1;
  uint32_t blocks = -1;
  uint32_t const sharedBytes = OPTIMAL_SMEM_BYTES;
  optimalOccupancyFromBlock(pathTraceMegakernel, sharedBytes, true, blocks,
                            threads);
  // ensure we cap threads to 64 (2048 -> 8 floats per thread)
  threads = min(64, threads);
  std::cout << "Computed ccupancy for pathTraceMegakernel SMEM: " << sharedBytes
            << " to be " << blocks << " blocks, " << threads
            << " threads per block" << std::endl;

  // init scene
  std::cout << "Allocating host and device resources" << std::endl;
  HostTriangleScene h_scene;
  std::vector<Light> h_lights;
  std::vector<Light> h_infiniteLights;
  std::vector<BSDF> h_bsdfs;
  DeviceCamera h_camera;
  cornellBox(&h_scene, &h_lights, &h_infiniteLights, &h_bsdfs, &h_camera);

  // device side scene
  TriangleSoup d_scene = triSoupFromTriangles(h_scene, h_bsdfs.size());
  BSDF* d_bsdfs = deviceBSDF(h_bsdfs);
  Light* d_lights = nullptr;
  Light* d_infiniteLights = nullptr;
  deviceLights(h_lights, h_infiniteLights, &d_lights, &d_infiniteLights);
  DeviceCamera* d_camera = deviceCamera(h_camera);
  DeviceHaltonOwen* d_rng = copyHaltonOwenToDeviceAlloc(blocks, threads);
  allocateDeviceConstantMemory(h_camera);

  // output buffers
  size_t const outputBytes = h_camera.width * h_camera.height * sizeof(float4);
#if DMT_ENABLE_MSE
  HostPinnedOutputBuffer h_out{};
  DeviceOutputBuffer d_out{};
  h_out.allocate(h_camera.width, h_camera.height);
  d_out.allocate(h_camera.width, h_camera.height);
#else
  float4* h_outputBuffer = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_outputBuffer, outputBytes));
  float4* d_outputBuffer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_outputBuffer,
                        h_camera.width * h_camera.height * sizeof(float4)));
#endif

  // stream
  cudaStream_t st_main = nullptr;
  CUDA_CHECK(cudaStreamCreate(&st_main));
  // timing
  AvgAndTotalTimer timer;

  static int constexpr MAX_SPP = 2048;  //
  std::cout << "Running CUDA Kernel" << std::endl;
  // TODO stream based write back
  for (uint32_t sTot = 0; sTot < MAX_SPP; sTot += h_camera.spp) {
    std::cout << "Running CUDA Kernel (" << sTot << ")" << std::endl;
#if DMT_ENABLE_MSE
    pathTraceMegakernel<<<blocks, threads, sharedBytes, st_main>>>(
        d_camera, d_scene, d_lights, h_lights.size(), d_infiniteLights,
        h_infiniteLights.size(), d_bsdfs, h_bsdfs.size(), sTot, d_rng, d_out);
#else
    pathTraceMegakernel<<<blocks, threads, sharedBytes, st_main>>>(
        d_camera, d_scene, d_lights, h_lights.size(), d_infiniteLights,
        h_infiniteLights.size(), d_bsdfs, h_bsdfs.size(), sTot, d_rng,
        d_outputBuffer);
#endif
    CUDA_CHECK(cudaGetLastError());

#if DMT_ENABLE_MSE
    CUDA_CHECK(cudaMemcpyAsync(h_out.m2Ptr, d_out.m2Ptr, outputBytes,
                               cudaMemcpyDeviceToHost, st_main));
    CUDA_CHECK(cudaMemcpyAsync(h_out.meanPtr, d_out.meanPtr, outputBytes,
                               cudaMemcpyDeviceToHost, st_main));
#else
    CUDA_CHECK(cudaMemcpyAsync(h_outputBuffer, d_outputBuffer, outputBytes,
                               cudaMemcpyDeviceToHost, st_main));
#endif
    CUDA_CHECK(cudaStreamSynchronize(st_main));

#if DMT_ENABLE_MSE
    std::string const name = "output-" + std::to_string(sTot + h_camera.spp);
#else
    std::string const name =
        "output-" + std::to_string(sTot + h_camera.spp) + ".png";
#endif
    std::cout << "Writing to file" << std::endl;
    timer.tick();

    // copy to host and to file (reset timer to not account for write to file)
#if DMT_ENABLE_MSE
    writeMeanAndMSERowMajor(h_out.meanPtr, h_out.m2Ptr, h_camera.width,
                            h_camera.height, name);
#else
    writeOutputBufferRowMajor(h_outputBuffer, h_camera.width, h_camera.height,
                              name.c_str());
#endif
    timer.reset();
  }
  std::cout << "Done! Total Execution Time(excl write file): "
            << timer.elapsedMillis()
            << " ms | Average Execution per Kernel launch (" << h_camera.spp
            << " spp): " << timer.avgMillis() << " ms" << std::endl;

  // cleanup
  std::cout << "Cleanup..." << std::endl;
  cudaStreamDestroy(st_main);

  freeDeviceConstantMemory();
#if DMT_ENABLE_MSE
#else
  cudaFree(h_outputBuffer);
  cudaFree(d_outputBuffer);
#endif
  cudaFree(d_scene.matId);
  cudaFree(d_scene.xs);
  cudaFree(d_scene.ys);
  cudaFree(d_scene.zs);
  cudaFree(d_bsdfs);
  cudaFree(d_rng);
  cudaFree(d_lights);
  cudaFree(d_infiniteLights);
  cudaFree(d_camera);
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
  megakernelMain();
}
