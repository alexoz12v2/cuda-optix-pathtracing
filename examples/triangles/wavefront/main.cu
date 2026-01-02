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
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

#include <cassert>

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

namespace cg = cooperative_groups;

// TODO: Optimize for vectorized load/store instructions
// ld.global.b128 16

// path state
// - pixel coords, sample index
// - samples per pixel
// - depth
// - throughput
// - transmission count
// - last was transmission
// - radiance
// - any specular bounces
// - last BSDF PDF
struct PathState {
  __device__ static PathState make(int px, int py, int s, int spp) {
    PathState state{};
    state.pixelCoordX = px;
    state.pixelCoordY = py;
    state.sampleIndex = s;
    state.spp = spp;
    state.depth = 0;
    state.throughput = make_float3(1, 1, 1);
    state.L = make_float3(0, 0, 0);
    state.lastBsdfPdf = 0;
    state.transmissionCount = 0;
    state.lastBouncetransmission = 0;
    state.anySpecularBounces = 0;
  }

  int pixelCoordX;
  int pixelCoordY;
  int sampleIndex;
  int spp;
  int depth;
  float3 throughput;
  float3 L;
  float lastBsdfPdf;
  int16_t transmissionCount;
  int8_t lastBouncetransmission;
  int8_t anySpecularBounces;
};
static_assert(sizeof(PathState) == 52);

struct RaygenInput {
  int px;
  int py;
  int sampleIndex;
  int spp;
  Transform const* cameraFromRaster;
  Transform const* renderFromCamera;
};

struct RaygenOutput {
  PathState* state;
  Ray ray;
};
static_assert(sizeof(RaygenOutput) == 76);

// Kernels:
// - raygen (pixel coords, sample index -> ray) (camera, RNG)
// - anyhit (ray, path state, scene) -> HitResult
//   - intersect
// - closesthit (ray, path state, scene) -> HitResult
//   - intersect
// - miss (path state)
// - shade (path state, HitResult)
struct WavefrontInput {
  DeviceCamera* d_cam;
  TriangleSoup d_triSoup;
  BSDF* d_bsdf;
  Light* d_lights;
  Light* d_infiniteLights;
  DeviceHaltonOwen* d_haltonOwen;
  float4* d_outBuffer;
  uint32_t bsdfCount;
  uint32_t lightCount;
  uint32_t infiniteLightCount;
  uint32_t sampleOffset;

  // GMEM Queues
  // raygen: input(S, px, py, s), output: (S, Ray)
  QueueGMEM<RaygenInput> raygenInput;
  QueueGMEM<RaygenOutput> raygenOutput;
};
// kernel param size limit (cc < 7.0)
static_assert(sizeof(WavefrontInput) <= 4096 * 1024);

__device__ RaygenOutput raygen(RaygenInput const& raygenInput,
                               DeviceHaltonOwen& warpRng,
                               DeviceHaltonOwenParams const& params) {
  int2 const pixel = make_int2(raygenInput.px, raygenInput.py);
  CameraSample const cs = getCameraSample(pixel, warpRng, params);
  RaygenOutput out{};
  PathState const state = PathState::make(raygenInput.px, raygenInput.py,
                              raygenInput.sampleIndex, raygenInput.spp);
  out.state = (PathState*)malloc(sizeof(PathState));
  memcpy(out.state, &state, sizeof(PathState));
  out.ray = getCameraRay(cs, *raygenInput.cameraFromRaster,
                         *raygenInput.renderFromCamera);
  return out;
}

__global__ void wavefrontKernel(WavefrontInput input) {
  // initialize camera ray 
}

namespace {

void optimalBlocksAndThreads(uint32_t& blocks, uint32_t& threads) {
  // do we actually support cooperative kernel launches? (Pascal: yes.)
  int supportsCoopLaunch = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch,
                                    cudaDevAttrCooperativeLaunch, 0));
  if (!supportsCoopLaunch) {
    std::cerr << "\033[31mCooperativeLaunch attribute not supported\033[0m"
              << std::endl;
    exit(1);
  }
  // ensure that blocks can be all resident in the GPU at once
  // - either use 1 block per SM (*)
  // - or use the Occupancy API to figure out max blocks
  cudaDeviceProp deviceProp{};
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  int const numKernels = 2;  // raygen, closesthit
  int const blocksPerKernel = 1;
  blocks = numKernels * blocksPerKernel;

  int const sharedBytes = 0;

#if 1
  int desiredThreads = 1;
#else
  int desiredThreads = 512;
#endif
  int availableBlocks = 0;
  while (availableBlocks == blocks && desiredThreads != 0) {
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &availableBlocks, wavefrontKernel, desiredThreads, sharedBytes));
    if (availableBlocks != blocks) {
      desiredThreads >>= 1;
    }
  }
  if (desiredThreads < WARP_SIZE || availableBlocks != blocks) {
    std::cerr << "\033[31mCooperativeLaunch blocks couldn't be satisfied\033[0m"
              << std::endl;
    exit(1);
  }

  assert(desiredThreads % WARP_SIZE == 0);
  threads = desiredThreads;
}

void wavefrontMain() {
  uint32_t threads = 0;
  uint32_t blocks = 0;
  optimalBlocksAndThreads(blocks, threads);
  std::cout << "Computed Optimal Occupancy for '2' kernels: " << blocks
            << " blocks " << threads << " threads" << std::endl;

  // init scene
  std::cout << "Allocating host and device resources" << std::endl;
  HostTriangleScene h_scene;
  std::vector<Light> h_lights;
  std::vector<Light> h_infiniteLights;
  std::vector<BSDF> h_bsdfs;
  DeviceCamera h_camera;
  cornellBox(&h_scene, &h_lights, &h_infiniteLights, &h_bsdfs, &h_camera);

  WavefrontInput kinput;
  kinput.d_triSoup = triSoupFromTriangles(h_scene, h_bsdfs.size());
  kinput.d_bsdf = deviceBSDF(h_bsdfs);
  kinput.bsdfCount = h_bsdfs.size();
  deviceLights(h_lights, h_infiniteLights, &kinput.d_lights,
               &kinput.d_infiniteLights);
  kinput.lightCount = h_lights.size();
  kinput.infiniteLightCount = h_infiniteLights.size();
  kinput.d_cam = deviceCamera(h_camera);
  kinput.d_haltonOwen = copyHaltonOwenToDeviceAlloc(blocks, threads);
  kinput.d_outBuffer = deviceOutputBuffer(h_camera.width, h_camera.height);

  allocateDeviceGGXEnergyPreservingTables();

  std::cout << "Cleanup..." << std::endl;
  cudaFree(kinput.d_outBuffer);
  cudaFree(kinput.d_haltonOwen);
  cudaFree(kinput.d_triSoup.matId);
  cudaFree(kinput.d_triSoup.xs);
  cudaFree(kinput.d_triSoup.ys);
  cudaFree(kinput.d_triSoup.zs);
  cudaFree(kinput.d_bsdf);
  freeDeviceGGXEnergyPreservingTables();
  cudaFree(kinput.d_lights);
  cudaFree(kinput.d_infiniteLights);
  cudaFree(kinput.d_cam);
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
#if 0
  wavefrontMain();
#endif
}
