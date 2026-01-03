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
#include <device_types.h>

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

struct IntersectionInput {
  PathState* state;
  Ray ray;
};
static_assert(sizeof(IntersectionInput) == 32);

struct ShadeInput {
  PathState* state;
  float3 pos;
  float3 normal;
  float3 error;    // intersection error bounds
  uint32_t matId;  // bsdf index
  float t;
  uint32_t _padding;
};
static_assert(sizeof(ShadeInput) == 56);
struct MissInput {
  PathState* state;
  float3 rayDirection;
};
static_assert(sizeof(MissInput) == 24);

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
  uint32_t* signalTerm;
  // GMEM Queues
  QueueGMEM<IntersectionInput> closesthitQueue;
  QueueGMEM<IntersectionInput> anyhitQueue;
  QueueGMEM<MissInput> missQueue;
  QueueGMEM<ShadeInput> shadeQueue;
};
// kernel param size limit (cc < 7.0)
static_assert(sizeof(WavefrontInput) <= 4096 * 1024);

__device__ IntersectionInput raygen(RaygenInput const& raygenInput,
                                    DeviceHaltonOwen& warpRng,
                                    DeviceHaltonOwenParams const& params) {
  int2 const pixel = make_int2(raygenInput.px, raygenInput.py);
  CameraSample const cs = getCameraSample(pixel, warpRng, params);
  IntersectionInput out{};
  PathState const state = PathState::make(
      raygenInput.px, raygenInput.py, raygenInput.sampleIndex, raygenInput.spp);
  out.state = static_cast<PathState*>(malloc(sizeof(PathState)));
  memcpy(out.state, &state, sizeof(PathState));
  out.ray = getCameraRay(cs, *raygenInput.cameraFromRaster,
                         *raygenInput.renderFromCamera);
  return out;
}

// height: 128, width: 128, spp: 4
// gridDim: 2
// loop 0:
// - b0: (x:0,  y:0) - (x:7,    y:3) | s: 0-3
// - b1: (x:8,  y:0) - (x:15,   y:3) | s: 0-3
// loop 1:
// - b0: (x:16,  y:0) - (x:23,  y:3) | s: 0-3
// - b1: (x:24,  y:3) - (x:31,  y:3) | s: 0-3
// height: 8, width: 8, spp: 8
// gridDim: 2
// loop 0:
// - b0: (x:0,   y:0) - (x:7,    y:3) | s: 0-3
// - b1: (x:0,   y:0) - (x:7,    y:3) | s: 4-7
// loop 1:
// - b0: (x:0,   y:4) - (x:7,    y:7) | s: 0-3
// - b1: (x:0,   y:4) - (x:7,    y:7) | s: 4-7
// inputs: height, width, spp, warpId, blockId, gridDim, warpRayIdx
// output: xStart, xEnd, yStart, yEnd, sStart, sEnd
/// \warning sample stride = warpSize
/*
    static int constexpr samplesPerGen = 1; // -> Tx
    static int constexpr pxPerGen = 8;      // -> Ty
    static int constexpr pyPerGen = 4;      // -> Tz
*/

inline constexpr int WAVEFRONT_KERNEL_TYPES = 5;

__constant__ float CMEM_cameraFromRaster[32];
__constant__ float CMEM_renderFromCamera[32];
__constant__ DeviceHaltonOwenParams CMEM_haltonOwenParams;

inline __host__ __device__ __forceinline__ Transform const* arrayAsTransform(
    float const* arr) {
  static_assert(sizeof(Transform) == 32 * sizeof(float) &&
                alignof(Transform) <= 16);
  // 16-byte aligned and at least 32 elements
  return reinterpret_cast<Transform const*>(arr);
}

// should be called from raygen block
// Bx: spp, By: width, Bz: height.
// int const S = spp;       // total samples per pixel
// int const W = width;     // image width
// int const H = height;    // image height
// int const ps = psPerGen; // samples per warp iteration
// int const px = pxPerGen; // 8
// int const py = pyPerGen; // 4
//
// global index = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
// blockIdx.x)
//                |-    block Number -|
//                * (blockDim.x * blockDim.y * blockDim.y)
//                |-   how many elements does a block have -|
//                + (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y *
// blockDim.x + threadIdx.x)
//                |- thread offset inside selected block -|
//
// warpRayIdx -> slice 32 elements linearized
// 3D grid indexing (example for ns: 2, nw: 2, nh: 2)
// s0 w0 h0 | s1 w0 h0 | s0 w1 h0 | s1 w1 h0 | s0 w2 h0 | s1 w2 h0 |
// s0 w0 h1 | s1 w0 h1 | s0 w1 h1 | s1 w1 h1 | s0 w2 h1 | s1 w2 h1 |
__device__ __forceinline__ void raygenPositionFromWarp(int H, int W, int S,
                                                       int linear, int& xStart,
                                                       int& xEnd, int& yStart,
                                                       int& yEnd, int& sStart,
                                                       int& sEnd) {
  static int constexpr nSamplesLoop = 4;
  static int constexpr samplesPerGen = 1;
  static int constexpr ps = nSamplesLoop * samplesPerGen;  // -> Tx
  static int constexpr px = 8;                             // -> Ty
  static int constexpr py = 4;                             // -> Tz
  static_assert(samplesPerGen * px * py == WARP_SIZE);

  int gridS = ceilDiv(S, ps);  // sample tiles
  int gridX = ceilDiv(W, px);  // pixel tiles in X
  int gridY = ceilDiv(H, py);  // pixel tiles in Y

  int sTile = linear % gridS;
  int xTile = (linear / gridS) % gridX;
  int yTile = linear / (gridS * gridX);

  sStart = sTile * ps;
  sEnd = min(sStart + ps, S);
  xStart = xTile * px;
  xEnd = min(xStart + px, W);
  yStart = yTile * py;
  yEnd = min(yStart + py, H);
}

inline constexpr int WAVEFRONT_KERNEL_RAYGEN_DIVISOR = 0;
inline constexpr int WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR = 1;
inline constexpr int WAVEFRONT_KERNEL_ANYHIT_DIVISOR = 2;
inline constexpr int WAVEFRONT_KERNEL_MISS_DIVISOR = 3;
inline constexpr int WAVEFRONT_KERNEL_SHADE_DIVISOR = 4;

// Dynamic SMEM: 8B
__global__ __launch_bounds__(512, WAVEFRONT_KERNEL_TYPES)
    __maxnreg__(32) void wavefrontKernel(WavefrontInput input) {
  // TODO init
  // assumes 1D grid
  extern __shared__ int SMEM[];
  int const blockType = blockIdx.x % WAVEFRONT_KERNEL_TYPES;
  // assumes 1D block
  assert(blockDim.x % warpSize == 0);

  using ThisWarp = cg::thread_block_tile<32>;
  ThisWarp theWarp = cg::tiled_partition<32>(cg::this_thread_block());

  DeviceHaltonOwen& warpRng =
      input.d_haltonOwen[(blockIdx.x * blockDim.x + threadIdx.x) / warpSize];

  if (blockType == WAVEFRONT_KERNEL_RAYGEN_DIVISOR) {
    static int constexpr ps = 4;
    static int constexpr px = 8;
    static int constexpr py = 4;
    static_assert(py * px == WARP_SIZE);
    int& raysPerBlock = SMEM[0];
    int& blockWideRayIndex = SMEM[1];

    if (threadIdx.x == 0) {
      blockWideRayIndex = 0;
      int const gridS = ceilDiv(input.d_cam->spp, ps);
      int const gridX = ceilDiv(input.d_cam->width, px);
      int const gridY = ceilDiv(input.d_cam->height, py);
      int const totalTiles = gridS * gridX * gridY;
      // TODO check PTX vectorized ld.global.b128 instruction
      raysPerBlock = ceilDiv(gridS * gridX * gridY,
                             (int)gridDim.x / WAVEFRONT_KERNEL_TYPES);
      assert(raysPerBlock % warpSize == 0);
    }
    __syncthreads();

    // while block is not done
    int ticket = 0;
    while ((ticket = atomicAggInc(&raysPerBlock)) < raysPerBlock) {
      assert(__activemask() == 0xFFFF'FFFFU);
      static constexpr int wkt2 =
          WAVEFRONT_KERNEL_TYPES * WAVEFRONT_KERNEL_TYPES;
      int const linear = blockIdx.x * blockDim.x / (wkt2 * warpSize) + ticket;
      uint32_t const lane =
          ThisWarp::thread_rank();  // TODO mov.u32 %r1, %laneid;

      int xStart = 0;
      int xEnd = 0;
      int yStart = 0;
      int yEnd = 0;
      int sStart = 0;
      int sEnd = 0;
      if (ThisWarp::thread_rank() == 0) {
        raygenPositionFromWarp(input.d_cam->height, input.d_cam->width,
                               input.d_cam->spp, linear, xStart, xEnd, yStart,
                               yEnd, sStart, sEnd);
        sStart += input.sampleOffset;
        sEnd += input.sampleOffset;
      }
      ThisWarp::sync();
      theWarp.shfl(xStart, 0);
      theWarp.shfl(xEnd, 0);
      theWarp.shfl(yStart, 0);
      theWarp.shfl(yEnd, 0);
      theWarp.shfl(sStart, 0);
      theWarp.shfl(sEnd, 0);

      int const sx = lane % ps;          // sample within tile
      int const pxl = (lane / ps) % px;  // x offset
      int const pyl = lane / (ps * px);  // y offset
      int const x = xStart + pxl;
      int const y = yStart + pyl;
      int sample = sStart + sx;
      for (int ss = 0; ss < ps; ++ss) {
        sample += ss;

        // construct raygen input
        IntersectionInput raygenElem{};
        {
          RaygenInput const raygenInput = {
              .px = x,
              .py = y,
              .sampleIndex = sample,
              .spp = input.d_cam->spp,
              .cameraFromRaster = arrayAsTransform(CMEM_cameraFromRaster),
              .renderFromCamera = arrayAsTransform(CMEM_renderFromCamera),
          };
          warpRng.startPixelSample(CMEM_haltonOwenParams, make_int2(x, y),
                                   sample);
          raygenElem = raygen(raygenInput, warpRng, CMEM_haltonOwenParams);
        }
        // check if 32 available places in queue
        int mask = 0;
        while (!mask) {
          // TODO wait N clocks before retrying
          while (input.closesthitQueue.producerSize() < warpSize) {
            // busy wait
          }
          // warp-wide push
          assert(__activemask() == 0xFFFF'FFFFU);
          mask = input.closesthitQueue.push(&raygenElem);
          assert(mask == 0xFFFF'FFFFU || mask == 0);
        }
      }
    }
  } else if (blockType == WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR) {
    int& blockTerminated = SMEM[0];
    if (threadIdx.x == 0) {
      blockTerminated = 0;
    }
    __syncthreads();

    while (true) {
      assert(__activemask() == 0xFFFF'FFFFU);
      if (input.closesthitQueue.consumerSize() < warpSize) {
        assert(__activemask() == 0xFFFF'FFFFU);
        if (threadIdx.x == 0) {
          // TODO termination signal (block leader checks atomic var)
          blockTerminated = atomicAdd(input.signalTerm, 0);
        }
        __syncthreads();
        if (blockTerminated) {
          // if there are remainders which are less than warp size, let first
          // warp of the first block handle that, while everybody else dies
          uint32_t warpId = -1;
          asm volatile("mov.u32  %0, %%warpid;" : "=r"(warpId));
          if (blockIdx.x == WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR &&
              warpId == 0 && input.closesthitQueue.consumerSize() > 0) {
            // go down
          } else {
            break;
          }
        } else {
          // TODO wait N clocks before retrying
          continue;
        }
      }
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      IntersectionInput kinput{};
      int const mask = input.closesthitQueue.pop(&kinput);
      assert((!blockTerminated && (mask == 0xFFFF'FFFFU || mask == 0)) ||
             (blockTerminated && (mask != 0xFFFF'FFFFU && mask != 0)));
            
      cg::coalesced_group const group = cg::coalesced_threads();
      // Compute HitResult
      // if hit -> enqueue anyhitQueue
      // if not hit -> enqueue missQueue
      // group sync
    }
  } else if (blockType == WAVEFRONT_KERNEL_ANYHIT_DIVISOR) {
  } else if (blockType == WAVEFRONT_KERNEL_MISS_DIVISOR) {
  } else if (blockType == WAVEFRONT_KERNEL_SHADE_DIVISOR) {
    uint32_t warpId = -1;
    asm volatile("mov.u32  %0, %%warpid;" : "=r"(warpId));
    // warp-loop over BSDF types. First loop search for 32 ready queue, second
    // loop anything is fine
  }
}

namespace {

void optimalBlocksAndThreads(uint32_t& blocks, uint32_t& threads,
                             uint32_t& sharedBytes) {
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

  sharedBytes = 8;

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
  uint32_t sharedBytes = 0;
  optimalBlocksAndThreads(blocks, threads, sharedBytes);
  std::cout << "Computed Optimal Occupancy for '2' kernels: " << blocks
            << " blocks " << threads << " threads" << std::endl;

  // init scene
  std::cout << "Allocating host and device resources" << std::endl;
  HostTriangleScene h_scene;
  std::vector<Light> h_lights;
  std::vector<Light> h_infiniteLights;
  std::vector<BSDF> h_bsdfs;
  DeviceCamera h_camera;
  cornellBox(false, &h_scene, &h_lights, &h_infiniteLights, &h_bsdfs,
             &h_camera);

  WavefrontInput kinput{};
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
