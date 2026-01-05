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

// TODO: add device LTO

#define PRINT_ASSERT 1

__device__ void pascal_fixed_sleep(uint64_t nanoseconds) {
  uint64_t start;
  // Read the 64-bit nanosecond global timer
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));

  uint64_t now = start;
  while (now < start + nanoseconds) {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now));
  }
}

// TODO how to use?
__device__ float groupedAtomicFetch(float* address) {
  unsigned const fullwarp = __activemask();
#if __CUDA_ARCH__ >= 700
  // 1. Identify all lanes in the wapr hitting same address
  unsigned const mask = __match_any_sync(fullwarp, (uintptr_t)address);
  // 2. Elect a leader for each unique address group
  int const leader = __ffs(mask) - 1;
  int const laneId = threadIdx.x % 32;  // could use %%laneid;
  float val = (float)0;
  // 3. Only leader performs hardware atomic fetch
  if (laneId == leader) {
    val = atomicAdd(address, (float)0);
  }
  // 4. Broadcast the fetched value from the leader to all lanes in the sam
  // group. we use the same mask to ensure shuffle stays within the matched
  // group
  return __shfl_sync(mask, val, leader);
#else
  // loop through all possible groups emulating match with ballot instruction
  unsigned active = fullwarp;
  float result = (float)0;
  while (active > 0) {
    // 1. Pick the first remaining lane as the leader of the round
    int const leader = __ffs(active) - 1;
    // 2. Broadcast the leader's address to se who else matches it
    float* refAddr = (float*)__shfl_sync(active, (uintptr_t)address, leader);
    unsigned match = __ballot_sync(active, address == refAddr);
    // 3. leader performs atomic fetch
    float fetchedVal = (float)0;
    if ((threadIdx.x % 32) == leader) {
      fetchedVal = atomicAdd(refAddr, (float)0);
    }
    // 4. Broadcast value to everyone in the current match group from
    // currently elected leader.
    float const sharedVal = __shfl_sync(active, fetchedVal, leader);
    // 5. if current lane was part of group, then save result
    if (address == refAddr) {
      result = sharedVal;
    }
    // 6. Clear processed lanes
    active &= ~match;
  }
  return result;
#endif
}

// TODO: how can We group atomic operation per lanes sharing address
// __match_any_sync and __match_all_sync perform a broadcast-and-compare
// operation of a variable between threads within a warp. Supported by
// devices of compute capability 7.x or higher.
template <typename T>
  requires((std::is_floating_point_v<T> || std::is_integral_v<T>) &&
           sizeof(T) == 4)
__device__ void groupedAtomicAdd(T* address, T val) {
  unsigned const fullwarp = __activemask();
#if __CUDA_ARCH__ >= 700
  // 1. find all lanes that have the same address
  unsigned const mask = __match_any_sync(fullwarp, (uintptr_t)address);
  // 2. Identify the leader (lane with lowest ID in mask)
  int const leader = __ffs(mask) - 1;
  int const laneId = threadIdx.x % 32;  // might be %laneid
  // 3. Intra-warp reduction within group
  float res = val;  // start with your value
  for (int i = 0; i < 32; ++i) {
    // if not leader and active
    if (i != leader && (mask & (1 << i))) {
      res += __shfl_sync(mask, val, i);
    }
  }

  // 4. leader does the add
  if (laneId == leader) {
    atomicAdd(address, res);
  }
#else
  // pascal doesn't have match instructions. iterative ballot approach.
  // loop through unique addresses present in the warp one by one
  // 0. mask of lanes which haven't been processed yet
  unsigned active = fullwarp;
  while (active > 0) {
    // 1. Pick a reference address from the first active lane
    int const leader = __ffs(active) - 1;
    auto* const refAddr =
        (float*)__shfl_sync(active, (uintptr_t)address, leader);
    // 2. Find all other lanes with same address
    unsigned const matching = __ballot_sync(active, address == refAddr);
    // 3. reduction within the matching group
    float res = val;  // start with your value
    for (int i = 16; i > 0; i >>= 1) {
      float const temp = __shfl_down_sync(matching, res, i);
      // if lane is part of match, add value (%laneid might be used here)
      if (matching & (1 << ((threadIdx.x % 32) + i))) {
        res += temp;
      }
    }
    // 4. Leader of matching group performs the atomic
    if ((threadIdx.x % 32) == leader) {
      atomicAdd(refAddr, res);
    }
    // 5. Remove the processed lanes from the active mask
    active &= ~matching;
  }
#endif
}

// TODO __CUDA_ARCH__ >= 700 version
__device__ int groupedAtomicIncLeaderOnly(int* address) {
  // pascal doesn't have match instructions. iterative ballot approach.
  // loop through unique addresses present in the warp one by one
  // 0. mask of lanes which haven't been processed yet
  unsigned active = __activemask();
  int old = 0;
  while (active > 0) {
    // 1. Pick a reference address from the first active lane
    int const leader = __ffs(active) - 1;
    auto* const refAddr = (int*)__shfl_sync(active, (uintptr_t)address, leader);
    // 2. Find all other lanes with same address
    unsigned const matching = __ballot_sync(active, address == refAddr);
    // 3. Leader of matching group performs the atomic inc
    if ((threadIdx.x % 32) == leader) {
      old = atomicAdd(refAddr, 1);
    }
    // 4. Leader broadcasts result to members of the current matching group
    if (address == refAddr) {
      old = __shfl_sync(active, old, leader);
    }
    // 5. Remove the processed lanes from the active mask
    active &= ~matching;
  }
  return old;
}

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
    state.lastBounceTransmission = 0;
    state.anySpecularBounces = 0;
    return state;
  }

  int pixelCoordX;
  int pixelCoordY;
  int sampleIndex;
  int spp;
  int depth;
  // TODO padding around for vector atomic exchange?
  float3 throughput;
  float3 L;
  float lastBsdfPdf;
  int transmissionCount;
  int lastBounceTransmission;  // TODO cache in SMEM?
  int anySpecularBounces;
  int _padding;
};
static_assert(sizeof(PathState) == 64);
static_assert(offsetof(PathState, L) % 16 == 0);
__device__ __forceinline__ float3 atomicFetchBeta(PathState& state) {
  float3 beta{1, 1, 1};
  beta.x = groupedAtomicFetch(reinterpret_cast<float*>(&state.throughput) + 0);
  beta.y = groupedAtomicFetch(reinterpret_cast<float*>(&state.throughput) + 1);
  beta.z = groupedAtomicFetch(reinterpret_cast<float*>(&state.throughput) + 2);
  return beta;
}
// TODO preallocated?
__device__ __forceinline__ void freeState(PathState* state) { free(state); }

struct RaygenInput {
  int px;
  int py;
  int sampleIndex;
  int spp;
  Transform const* cameraFromRaster;
  Transform const* renderFromCamera;
};

struct ClosestHitInput {
  PathState* state;
  Ray ray;
};
static_assert(sizeof(ClosestHitInput) == 32);
struct AnyhitInput {
  PathState* state;
  float3 pos;
  float3 rayD;
  float3 normal;
  float3 error;    // intersection error bounds
  uint32_t matId;  // bsdf index
  float t;
};
static_assert(sizeof(AnyhitInput) == 64);

struct ShadeInput {
  PathState* state;
  float3 pos;
  float3 rayD;
  float3 normal;
  float3 error;    // intersection error bounds
  uint32_t matId;  // bsdf index
  float t;
};
static_assert(sizeof(ShadeInput) == 64);
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
  QueueGMEM<ClosestHitInput> closesthitQueue;
  QueueGMEM<AnyhitInput> anyhitQueue;
  QueueGMEM<MissInput> missQueue;
  QueueGMEM<ShadeInput> shadeQueue;
};
// kernel param size limit (cc < 7.0)
static_assert(sizeof(WavefrontInput) <= 4096 * 1024);

__device__ ClosestHitInput raygen(RaygenInput const& raygenInput,
                                  DeviceHaltonOwen& warpRng,
                                  DeviceHaltonOwenParams const& params) {
  int2 const pixel = make_int2(raygenInput.px, raygenInput.py);
  CameraSample const cs = getCameraSample(pixel, warpRng, params);
  ClosestHitInput out{};
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
__constant__ int2 CMEM_imageResolution;
__constant__ int CMEM_spp;

inline __host__ __device__ __forceinline__ Transform const* arrayAsTransform(
    float const* arr) {
  static_assert(sizeof(Transform) == 32 * sizeof(float) &&
                alignof(Transform) <= 16);
  // 16-byte aligned and at least 32 elements
  return reinterpret_cast<Transform const*>(arr);
}

// ----------------------------------------------------------------------------
// raygen
// ----------------------------------------------------------------------------

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

  int const gridS = ceilDiv(S, ps);  // sample tiles
  int const gridX = ceilDiv(W, px);  // pixel tiles in X
  // int const gridY = ceilDiv(H, py);  // pixel tiles in Y

  int const sTile = linear % gridS;
  int const xTile = (linear / gridS) % gridX;
  int const yTile = linear / (gridS * gridX);

  sStart = sTile * ps;
  sEnd = min(sStart + ps, S);
  xStart = xTile * px;
  xEnd = min(xStart + px, W);
  yStart = yTile * py;
  yEnd = min(yStart + py, H);
}

// ----------------------------------------------------------------------------
// closesthit
// ----------------------------------------------------------------------------

// TODO Compare __ldg with normal accesses

__device__ HitResult closesthit(Ray const& __restrict__ ray,
                                TriangleSoup const& __restrict__ triSoup) {
  HitResult result;
  for (int tri = 0; tri < triSoup.count; ++tri) {
    float4 const x = __ldg(reinterpret_cast<float4 const*>(triSoup.xs) + tri);
    float4 const y = __ldg(reinterpret_cast<float4 const*>(triSoup.ys) + tri);
    float4 const z = __ldg(reinterpret_cast<float4 const*>(triSoup.zs) + tri);
    HitResult const other = triangleIntersect(x, y, z, ray);
    if (other.hit && other.t < result.t) {
      result = other;
      // TODO better? Coalescing?
      result.matId = triSoup.matId[tri];
    }
  }
  return result;
}

// ----------------------------------------------------------------------------
// anyhit
// ----------------------------------------------------------------------------
__device__ void anyhitNEE(
    float3 const& __restrict__ wo, float3 const& __restrict__ beta,
    float3 const& __restrict__ surfPos, float3 const& __restrict__ surfPosError,
    BSDF const& __restrict__ bsdf, Light const& __restrict__ light,
    TriangleSoup const& __restrict__ triSoup, float3 const& __restrict__ normal,
    int const lightCount, bool lastBounceTransmission, float2 u,
    float3* __restrict__ Le) {
  if (LightSample const ls =
          sampleLight(light, surfPos, u, lastBounceTransmission, normal)) {
    Ray const ray{
        .o = offsetRayOrigin(surfPos, surfPosError, normal, ls.direction),
        .d = ls.direction};

    float bsdfPdf = 0;
    float3 const bsdf_f =
        evalBsdf(bsdf, wo, ray.d, normal, normal, &bsdfPdf) * bsdf.weight();
    if (!bsdfPdf) {
      return;
    }

#if PRINT_ASSERT
    if (isZero(bsdf_f)) {
      printf("[%u %u %d] !isZero(bsdf_f)\n", blockIdx.x, threadIdx.x, __LINE__);
    }
#endif
    assert(!isZero(bsdf_f));

    for (int tri = 0; tri < triSoup.count; ++tri) {
      float4 const x = __ldg(reinterpret_cast<float4 const*>(triSoup.xs) + tri);
      float4 const y = __ldg(reinterpret_cast<float4 const*>(triSoup.ys) + tri);
      float4 const z = __ldg(reinterpret_cast<float4 const*>(triSoup.zs) + tri);
      if (HitResult const result = triangleIntersect(x, y, z, ray);
          result.hit && result.t < ls.distance) {
        return;
      }
    }
    // not occluded
    *Le = evalLight(light, ls);
    if (ls.delta) {
      *Le = beta * *Le * bsdf_f * lightCount;
    } else {
      // MIS if not delta (and not BSDF Delta TODO)
      // power heuristic
      float const w = sqrf(10 / lightCount * ls.pdf) /
                      sqrf(10 / lightCount * ls.pdf + bsdfPdf);
      *Le = *Le * bsdf_f * beta * w;
    }
  }
}

// ----------------------------------------------------------------------------
// wavefront kernel
// ----------------------------------------------------------------------------

inline constexpr int WAVEFRONT_KERNEL_RAYGEN_DIVISOR = 0;
inline constexpr int WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR = 1;
inline constexpr int WAVEFRONT_KERNEL_ANYHIT_DIVISOR = 2;
inline constexpr int WAVEFRONT_KERNEL_MISS_DIVISOR = 3;
inline constexpr int WAVEFRONT_KERNEL_SHADE_DIVISOR = 4;

// only one of __lanuch_bounds__ and __maxnreg__ can be applied to a kernel
// (max reg count can be manipulated from nvcc)

// Dynamic SMEM: 12B
__global__ __launch_bounds__(512, WAVEFRONT_KERNEL_TYPES)
    //__maxnreg__(32)
    void wavefrontKernel(WavefrontInput input) {
  // TODO init
  // assumes 1D grid
  extern __shared__ int SMEM[];
  int const blockType = blockIdx.x % WAVEFRONT_KERNEL_TYPES;
  int const blockNumPerKernel = gridDim.x / WAVEFRONT_KERNEL_TYPES;
  // assumes 1D block
#if PRINT_ASSERT
  if (blockDim.x % warpSize != 0) {
    printf("[%u %u %d] blockDim.x %% warpSize == 0\n", blockIdx.x, threadIdx.x,
           __LINE__);
  }
#endif
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
    int& warpRemaining = SMEM[2];
    // 4 byte per warp (note: other kernel types will use SMEM queue caches)

    if (threadIdx.x == 0) {
      blockWideRayIndex = 0;
      int const gridS = ceilDiv(CMEM_spp, ps);
      int const gridX = ceilDiv(input.d_cam->width, px);
      int const gridY = ceilDiv(input.d_cam->height, py);
      // TODO check PTX vectorized ld.global.b128 instruction
      raysPerBlock =
          ceilDiv(gridS * gridX * gridY,
                  static_cast<int>(gridDim.x) / WAVEFRONT_KERNEL_TYPES);
#if PRINT_ASSERT
      if (raysPerBlock % warpSize != 0) {
        printf("[%u %u %d] raysPerBlock %% warpSize == 0\n", blockIdx.x,
               threadIdx.x, __LINE__);
      }
#endif
      assert(raysPerBlock % warpSize == 0);

      if (blockIdx.x == WAVEFRONT_KERNEL_RAYGEN_DIVISOR && threadIdx.x == 0) {
        int nwarpId;
        asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(nwarpId));
        warpRemaining = nwarpId;
      }
    }
    __syncthreads();
    static constexpr int wkt2 = WAVEFRONT_KERNEL_TYPES * WAVEFRONT_KERNEL_TYPES;

    // while block is not done
    int ticket = 0;
    while ((ticket = atomicAggInc(&blockWideRayIndex)) < raysPerBlock) {
#if PRINT_ASSERT
      if (__activemask() != 0xFFFF'FFFFU) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU\n", blockIdx.x,
               threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU);
      uint32_t const lane = ThisWarp::thread_rank();
      bool lastTicket = false;
      if (lane == 0) {
#if PRINT_ASSERT
        if ((ticket & 31) != 0) {
          printf("[%u %u %d] (ticket & 31) == 0\n", blockIdx.x, threadIdx.x,
                 __LINE__);
        }
#endif
        assert((ticket & 31) == 0);
        // last warp of raygen block: 1, ...
        const int raygenWarpIdxCompl =
            blockDim.x * (gridDim.x - blockIdx.x) / (wkt2 * warpSize) + 1;
        lastTicket = ticket + raygenWarpIdxCompl * warpSize >= raysPerBlock;
      }
      ThisWarp::sync();
      lastTicket = theWarp.shfl(lastTicket, 0);

      int xStart = 0;
      int xEnd = 0;
      int yStart = 0;
      int yEnd = 0;
      int sStart = 0;
      int sEnd = 0;
      if (ThisWarp::thread_rank() == 0) {
        int const linear = blockIdx.x * blockDim.x / (wkt2 * warpSize) + ticket;
        raygenPositionFromWarp(CMEM_imageResolution.y, CMEM_imageResolution.x,
                               CMEM_spp, linear, xStart, xEnd, yStart, yEnd,
                               sStart, sEnd);
        sStart += input.sampleOffset;
        sEnd += input.sampleOffset;
      }
      ThisWarp::sync();
      xStart = theWarp.shfl(xStart, 0);
      xEnd = theWarp.shfl(xEnd, 0);
      yStart = theWarp.shfl(yStart, 0);
      yEnd = theWarp.shfl(yEnd, 0);
      sStart = theWarp.shfl(sStart, 0);
      sEnd = theWarp.shfl(sEnd, 0);

      int const x = xStart + (lane / ps) % px;
      int const y = yStart + lane / (ps * px);
      for (int ss = 0; ss < ps; ++ss) {
        int const sample = (sStart + lane % ps) + ss;

        // construct raygen input
        ClosestHitInput raygenElem{};  // TODO SMEM?
        {
          // TODO without struct
          RaygenInput const raygenInput = {
              .px = x,
              .py = y,
              .sampleIndex = sample,
              .spp = input.d_cam->spp,
              .cameraFromRaster = arrayAsTransform(CMEM_cameraFromRaster),
              .renderFromCamera = arrayAsTransform(CMEM_renderFromCamera),
          };
          // TODO add start pixel sample to every kernel
          warpRng.startPixelSample(CMEM_haltonOwenParams, make_int2(x, y),
                                   sample);
          raygenElem = raygen(raygenInput, warpRng, CMEM_haltonOwenParams);
          // we got here "AFTER raygen: 0xffffffff"
          // printf("AFTER raygen: 0x%x\n", __activemask());
        }
        // check if 32 available places in queue
        while (input.closesthitQueue.producerFreeCount() < warpSize &&
               !lastTicket) {
          // busy wait
          pascal_fixed_sleep(64);
        }
        unsigned active = __activemask();
        while (active) {
          // warp-wide push
          active &= ~input.closesthitQueue.push(&raygenElem);
          if (!active) {
            pascal_fixed_sleep(64);
          }
        }
        // printf("RAYGEN activemask 0x%x generated sample %d\n",
        // __activemask(), sample);
      }  // sample loop
      ThisWarp::sync();
    }  // persistent kernel thread loop

#if PRINT_ASSERT
    if (__activemask() != 0xFFFF'FFFFu) {
      printf("[%u %u %d] __activemask() == 0xFFFF'FFFFu\n", blockIdx.x,
             threadIdx.x, __LINE__);
    }
#endif
    assert(__activemask() == 0xFFFF'FFFFu);
    // __ffs(__activemask()) - 1
    // warp leader decrements warpRemaining
    if ((threadIdx.x & 31) == 0) {
      // if warpRemaining zero (not negative),
      // warp leader increments GMEM signalTerm
      if (atomicSub(&warpRemaining, 1) == 0) {
        atomicInc(input.signalTerm, 1);
      }
    }

  } else if (blockType == WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR) {
    int& blockTerminated = SMEM[0];
    if (threadIdx.x == 0) {
      blockTerminated = 0;
    }
    __syncthreads();

    while (true) {
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      if (input.closesthitQueue.consumerUsedCount() < warpSize) {
#if PRINT_ASSERT
        if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
          printf(
              "[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
              blockIdx.x, threadIdx.x, __LINE__);
        }
#endif
        assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
        if (threadIdx.x == 0) {
          // termination signal (block leader checks atomic var)
          blockTerminated = atomicAdd(input.signalTerm, 0);
        }
        // atomic operations shouldn't need external synchronization. also, this
        // can cause deadlock
        // __syncthreads();
        if (blockTerminated == blockNumPerKernel) {
          // if there are remainders which are less than warp size, let first
          // warp of the first block handle that, while everybody else dies
          uint32_t warpId = -1U;
          asm volatile("mov.u32  %0, %%warpid;" : "=r"(warpId));
          if (blockIdx.x == WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR &&
              warpId == 0 && input.closesthitQueue.consumerUsedCount() > 0) {
            // go down
          } else {
            break;
          }
        } else {
          pascal_fixed_sleep(64);
          continue;
        }
      }
      // printf("CLOSEST HIT START \n");
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      ClosestHitInput kinput{};  // TODO SMEM?
      int const mask = input.closesthitQueue.pop(&kinput);
      if (!(mask & (1 << ThisWarp::thread_rank()))) {
#if PRINT_ASSERT
        if (!blockTerminated) {
          printf("[%u %u %d] blockTerminated\n", blockIdx.x, threadIdx.x,
                 __LINE__);
        }
#endif
        assert(blockTerminated);
        // this is the last batch
        break;
      }
      cg::coalesced_group const g = cg::coalesced_threads();

      // Compute HitResult
      HitResult const result = closesthit(kinput.ray, input.d_triSoup);
      if (result.hit) {
        // if hit -> enqueue anyhitQueue
        cg::coalesced_group const gHit = cg::coalesced_threads();
        // TODO SMEM?
        AnyhitInput const anyHitInput{.state = kinput.state,
                                      .pos = result.pos,
                                      .rayD = kinput.ray.d,
                                      .normal = result.normal,
                                      .error = result.error,
                                      .matId = result.matId,
                                      .t = result.t};
        int coalescedMask = 0;
        do {
          coalescedMask = input.anyhitQueue.push(&anyHitInput);
          if (!(coalescedMask & (1 << gHit.thread_rank()))) {
            pascal_fixed_sleep(64);
          }
        } while (!(coalescedMask & (1 << gHit.thread_rank())));

        // enqueue shadeQueue (TODO SMEM?)
        ShadeInput const shadeInput{
            .state = kinput.state,
            .pos = result.pos,
            .rayD = kinput.ray.d,
            .normal = result.normal,
            .error = result.error,
            .matId = result.matId,
            .t = result.t,
        };
        do {
          coalescedMask = input.shadeQueue.push(&shadeInput);
          if (!(coalescedMask & (1 << g.thread_rank()))) {
            pascal_fixed_sleep(64);
          }
        } while (!(coalescedMask & (1 << g.thread_rank())));

      } else {
        // if not hit -> enqueue missQueue
        cg::coalesced_group const gMiss = cg::coalesced_threads();
        // TODO SMEM?
        MissInput const missInput{.state = kinput.state,
                                  .rayDirection = kinput.ray.d};
        int coalescedMask = 0;
        do {
          coalescedMask = input.missQueue.push(&missInput);
          if (!(coalescedMask & (1 << gMiss.thread_rank()))) {
            pascal_fixed_sleep(64);
          }
        } while (!(coalescedMask & (1 << gMiss.thread_rank())));
      }
      // group sync
      g.sync();
    }
  } else if (blockType == WAVEFRONT_KERNEL_ANYHIT_DIVISOR) {
    // preamble same as closest hit
    int& blockTerminated = SMEM[0];
    if (threadIdx.x == 0) {
      blockTerminated = 0;
    }
    __syncthreads();

    while (true) {
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      if (input.anyhitQueue.consumerUsedCount() < warpSize) {
#if PRINT_ASSERT
        if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
          printf(
              "[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
              blockIdx.x, threadIdx.x, __LINE__);
        }
#endif
        assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
        if (threadIdx.x == 0) {
          // termination signal (block leader checks atomic var)
          blockTerminated = atomicAdd(input.signalTerm, 0);
        }
        // atomic operations shouldn't need external synchronization. also, this
        // can cause deadlock
        // __syncthreads();
        if (blockTerminated == blockNumPerKernel) {
          // if there are remainders which are less than warp size, let first
          // warp of the first block handle that, while everybody else dies
          uint32_t warpId = -1;
          asm volatile("mov.u32  %0, %%warpid;" : "=r"(warpId));
          if (blockIdx.x == WAVEFRONT_KERNEL_ANYHIT_DIVISOR && warpId == 0 &&
              input.anyhitQueue.consumerUsedCount() > 0) {
            // go down
          } else {
            break;
          }
        } else {  // NOT blockTerminated
          // TODO wait N clocks before retrying
          continue;
        }
      }
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      // TODO is this necessary?
      AnyhitInput kinput{};  // TODO SMEM?
      int const mask = input.anyhitQueue.pop(&kinput);
      if (!(mask & (1 << ThisWarp::thread_rank()))) {
#if PRINT_ASSERT
        if (!blockTerminated) {
          printf("[%u %u %d] blockTerminated\n", blockIdx.x, threadIdx.x,
                 __LINE__);
        }
#endif
        assert(blockTerminated);
        // this is the last batch
        break;
      }
      cg::coalesced_group const g = cg::coalesced_threads();
      // choose a light and prepare BSDF data from intersection
      // TODO maybe: readonly function to fetch 32 bytes with __ldg?
      Light const light = input.d_lights[min(  // TODO copy?
          static_cast<int>(warpRng.get1D(CMEM_haltonOwenParams) *
                           input.lightCount),
          input.lightCount - 1)];
      BSDF const bsdf = [&] __device__() {  // TODO copy?
        BSDF theBsdf = input.d_bsdf[kinput.matId];
        prepareBSDF(&theBsdf, kinput.normal, -kinput.rayD);
        return theBsdf;
      }();

      // cast a shadow ray and perform a anyhit query
      // if hit -> NEE
      float3 Le{0, 0, 0};
      float3 const beta = atomicFetchBeta(*kinput.state);
      anyhitNEE(-kinput.rayD, beta, kinput.pos, kinput.error, bsdf, light,
                input.d_triSoup, kinput.normal, input.lightCount,
                atomicAdd(&kinput.state->lastBounceTransmission, 0),
                warpRng.get2D(CMEM_haltonOwenParams), &Le);
      groupedAtomicAdd(reinterpret_cast<float*>(&kinput.state->L) + 0, Le.x);
      groupedAtomicAdd(reinterpret_cast<float*>(&kinput.state->L) + 1, Le.y);
      groupedAtomicAdd(reinterpret_cast<float*>(&kinput.state->L) + 2, Le.y);
      g.sync();
    }  // end persistent kernel thread loop
  } else if (blockType == WAVEFRONT_KERNEL_MISS_DIVISOR) {
    // usual consumer kernel preamble
    int& blockTerminated = SMEM[0];
    if (threadIdx.x == 0) {
      blockTerminated = 0;
    }
    __syncthreads();

    while (true) {
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      if (input.missQueue.consumerUsedCount() < warpSize) {
#if PRINT_ASSERT
        if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
          printf(
              "[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
              blockIdx.x, threadIdx.x, __LINE__);
        }
#endif
        assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
        if (threadIdx.x == 0) {
          // termination signal (block leader checks atomic var)
          blockTerminated = atomicAdd(input.signalTerm, 0);
        }
        // atomic operations shouldn't need external synchronization. also,
        // synch here can cause deadlocks
        // __syncthreads();
        if (blockTerminated == blockNumPerKernel) {
          // if there are remainders which are less than warp size, let first
          // warp of first block handle that, while everybody else dies
          uint32_t warpId = -1u;
          asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpId));
          if (blockIdx.x == WAVEFRONT_KERNEL_MISS_DIVISOR && warpId == 0 &&
              input.missQueue.consumerUsedCount() > 0) {
            // go down
          } else {
            break;
          }
        } else {  // NOT blockTerminated
          // TODO  wait N clocks before retrying
          continue;
        }
      }
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      MissInput kinput{};
      int const mask = input.missQueue.pop(&kinput);
      if (!(mask & (1 << ThisWarp::thread_rank()))) {
#if PRINT_ASSERT
        if (!blockTerminated) {
          printf("[%u %u %d] blockTerminated\n", blockIdx.x, threadIdx.x,
                 __LINE__);
        }
#endif
        assert(blockTerminated);
        // this is the last batch
        break;
      }
      cg::coalesced_group const g = cg::coalesced_threads();
      // 1. choose an infinite light, fetch last BSDF intersection (TODO)
      Light const light = input.d_infiniteLights[min(  // TODO copy?
          static_cast<int>(warpRng.get1D(CMEM_haltonOwenParams) *
                           input.infiniteLightCount),
          input.infiniteLightCount - 1)];
      // 2. add to state radiance
      float pdf = 0;
      float3 Le = evalInfiniteLight(light, kinput.rayDirection, &pdf);
      if (pdf) {
        float3 const beta = atomicFetchBeta(*kinput.state);
        Le = beta * Le * input.infiniteLightCount;
        // TODO: If any specular bounce or depth == 0, don't do MIS
        // TODO: Use last BSDF for MIS
        groupedAtomicAdd(&kinput.state->L.x, Le.x);
        groupedAtomicAdd(&kinput.state->L.y, Le.y);
        groupedAtomicAdd(&kinput.state->L.z, Le.z);
      }
      g.sync();
      // 3. sink to output buffer
      // no need for atomic operation here
      {
        int2 const pixel =
            __ldg(reinterpret_cast<int2*>(&kinput.state->pixelCoordX));
        int2 const imageRes = CMEM_imageResolution;
        // don't use ldg, as it's probably cached?
        float4 color = *reinterpret_cast<float4*>(&kinput.state->L);
        color.w = 1;  // TODO filter weight?
        // TODO SMEM?
        input.d_outBuffer[pixel.y * imageRes.x + pixel.x] += color;
      }
      freeState(kinput.state);
      kinput.state = nullptr;
    }  // end persistent thread loop
  } else if (blockType == WAVEFRONT_KERNEL_SHADE_DIVISOR) {
    // possible optimization:
    // - warp-loop over BSDF types. First loop search for 32 ready queue, second
    //   loop anything is fine
    int& blockTerminated = SMEM[0];
    if (threadIdx.x == 0) {
      blockTerminated = 0;
    }
    __syncthreads();
    while (true) {
#if PRINT_ASSERT
      if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
               blockIdx.x, threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
      if (input.shadeQueue.consumerUsedCount() < warpSize) {
#if PRINT_ASSERT
        if (!(__activemask() == 0xFFFF'FFFFU || blockTerminated)) {
          printf(
              "[%u %u %d] __activemask() == 0xFFFF'FFFFU || blockTerminated\n",
              blockIdx.x, threadIdx.x, __LINE__);
        }
#endif
        assert(__activemask() == 0xFFFF'FFFFU || blockTerminated);
        if (threadIdx.x == 0) {
          // termination signal (block leader checks atomic var)
          blockTerminated = atomicAdd(input.signalTerm, 0);
        }
        // atomic operations shouldn't need external synchronization. also, this
        // can cause deadlock
        // __syncthreads();
        if (blockTerminated == blockNumPerKernel) {
          // if there are remainders which are less than warp size, let first
          // warp of the first block handle that, while everybody else dies
          uint32_t warpId = -1u;
          asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpId));
          if (blockIdx.x == WAVEFRONT_KERNEL_SHADE_DIVISOR && warpId == 0 &&
              input.shadeQueue.consumerUsedCount() > 0) {
            // go down
          } else {
            break;
          }
        } else {
          // TODO wait N clocks before retrying
          continue;
        }
      }  // exit if-block without continue/break: then execute core function
#if PRINT_ASSERT
      if (__activemask() != 0xFFFF'FFFFU) {
        printf("[%u %u %d] __activemask() == 0xFFFF'FFFFU\n", blockIdx.x,
               threadIdx.x, __LINE__);
      }
#endif
      assert(__activemask() == 0xFFFF'FFFFU);
      ShadeInput kinput{};
      int const mask = input.shadeQueue.pop(&kinput);
      if (!(mask & (1 << ThisWarp::thread_rank()))) {
#if PRINT_ASSERT
        if (!blockTerminated) {
          printf("[%u %u %d] blockTerminated\n", blockIdx.x, threadIdx.x,
                 __LINE__);
        }
#endif
        assert(blockTerminated);
        // this is the last batch
        break;
      }
      cg::coalesced_group const g = cg::coalesced_threads();

      // Inside each bsdf
      // 1. if max depth reached, kill path
      bool pathDied = false;
      static int constexpr MAX_DEPTH = 32;
      int oldDepth = groupedAtomicIncLeaderOnly(&kinput.state->depth);
      if (oldDepth >= MAX_DEPTH) {
        pathDied = true;
      }

      float3 wi{0, 0, 0};
      if (!pathDied) {
        // 2. (not dead) BSDF sampling and bounce computation
        BSDF const bsdf = [&] __device__() {  // TODO copy?
          BSDF theBsdf = input.d_bsdf[kinput.matId];
          prepareBSDF(&theBsdf, kinput.normal, -kinput.rayD);
          return theBsdf;
        }();
        BSDFSample bs =
            sampleBsdf(bsdf, -kinput.rayD, kinput.normal, kinput.normal,
                       warpRng.get2D(CMEM_haltonOwenParams),
                       warpRng.get1D(CMEM_haltonOwenParams));
        if (bs) {
          wi = bs.wi;
          // 3. (not dead) update path state
          atomicAdd(&kinput.state->anySpecularBounces, (int)bs.delta);
          atomicAdd(&kinput.state->transmissionCount, (int)bs.refract);
          atomicExch(&kinput.state->lastBounceTransmission, (int)bs.refract);
          float3 beta = atomicFetchBeta(*kinput.state) * bs.f *
                        fabsf(dot(bs.wi, kinput.normal)) / bs.pdf;
          // 4. (not dead) russian roulette. If fails, kill path
          if (float const rrBeta = maxComponentValue(beta * bs.eta);
              rrBeta < 1 && oldDepth > 1) {
            float const q = fmaxf(0.f, 1.f - rrBeta);
            if (warpRng.get1D(CMEM_haltonOwenParams) < q) {
              pathDied = true;
            } else {
              beta /= 1 - q;
              atomicExch(&kinput.state->throughput.x, beta.x);
              atomicExch(&kinput.state->throughput.y, beta.y);
              atomicExch(&kinput.state->throughput.z, beta.z);
            }
          }
        } else {
          pathDied = true;
        }
      }

      if (pathDied) {
        // 5. if path killed, sink to output
        // no need for atomic operation here
        {
          int2 const pixel =
              __ldg(reinterpret_cast<int2*>(&kinput.state->pixelCoordX));
          int2 const imageRes = CMEM_imageResolution;
          // don't use ldg, as it's probably cached?
          float4 color = *reinterpret_cast<float4*>(&kinput.state->L);
          color.w = 1;  // TODO filter weight?
          // TODO SMEM?
          input.d_outBuffer[pixel.y * imageRes.x + pixel.x] += color;
        }
        freeState(kinput.state);
        kinput.state = nullptr;
      } else {
        // 6. if path alive, push to closesthit next bounce
        ClosestHitInput const closestHitInput{
            .state = kinput.state,
            .ray = {
                .o = offsetRayOrigin(kinput.pos, kinput.error, kinput.normal,
                                     wi),
                .d = wi,
            }};  // TODO SMEM?

        cg::coalesced_group const gAlive = cg::coalesced_threads();
        int coalescedMask = 0;
        do {
          coalescedMask = input.closesthitQueue.push(&closestHitInput);
          if (!(coalescedMask & (1 << gAlive.thread_rank()))) {
            pascal_fixed_sleep(64);
          }
        } while (!(coalescedMask & (1 << gAlive.thread_rank())));
      }
      g.sync();
    }  // end persistent kernel thread loop
  }
}

namespace {

__host__ void optimalBlocksAndThreads(uint32_t& blocks, uint32_t& threads,
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

  int const numKernels = WAVEFRONT_KERNEL_TYPES;  // raygen, closesthit
  int const blocksPerKernel = 1;
  // TODO dynamic sizing of blocks
  blocks = numKernels * blocksPerKernel;

  // To be kept in sync with kernel consumption
  sharedBytes = 12;

#if 1
  int desiredThreads = 32;
#else
  int desiredThreads = 512;
#endif
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

void allocateDeviceMemory(uint32_t threads, uint32_t blocks,
                          HostTriangleScene const& h_scene,
                          std::vector<Light> const& h_lights,
                          std::vector<Light> const& h_infiniteLights,
                          std::vector<BSDF> const& h_bsdfs,
                          DeviceCamera const& h_camera,
                          WavefrontInput& kinput) {
  // constant memory
  allocateDeviceGGXEnergyPreservingTables();
  {
    Transform const src =
        cameraFromRaster_Perspective(h_camera.focalLength, h_camera.sensorSize,
                                     h_camera.width, h_camera.height);
    cudaMemcpyToSymbol(CMEM_cameraFromRaster, &src, sizeof(Transform), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    DeviceHaltonOwenParams const params =
        DeviceHaltonOwen::computeParams(h_camera.width, h_camera.height);
    cudaMemcpyToSymbol(CMEM_haltonOwenParams, &params,
                       sizeof(DeviceHaltonOwenParams), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    int2 const imageRes = make_int2(h_camera.width, h_camera.height);
    cudaMemcpyToSymbol(CMEM_imageResolution, &imageRes, sizeof(int2), 0,
                       cudaMemcpyHostToDevice);
  }
  {
    Transform const src = worldFromCamera(h_camera.dir, h_camera.pos);
    cudaMemcpyToSymbol(CMEM_renderFromCamera, &src, sizeof(Transform), 0,
                       cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(CMEM_spp, &h_camera.spp, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  // output buffer
  kinput.d_outBuffer = nullptr;
  CUDA_CHECK(cudaMalloc(&kinput.d_outBuffer,
                        h_camera.width * h_camera.height * sizeof(float4)));
  CUDA_CHECK(cudaMemset(kinput.d_outBuffer, 0,
                        h_camera.width * h_camera.height * sizeof(float4)));
  // GMEM queues
  static int constexpr QUEUE_CAP = 1024;
  kinput.anyhitQueue = decltype(kinput.anyhitQueue)::create(QUEUE_CAP);
  kinput.closesthitQueue = decltype(kinput.closesthitQueue)::create(QUEUE_CAP);
  kinput.shadeQueue = decltype(kinput.shadeQueue)::create(QUEUE_CAP);
  kinput.missQueue = decltype(kinput.missQueue)::create(QUEUE_CAP);

  // scene
  kinput.d_triSoup = triSoupFromTriangles(h_scene, h_bsdfs.size());
  kinput.d_bsdf = deviceBSDF(h_bsdfs);
  kinput.bsdfCount = h_bsdfs.size();
  deviceLights(h_lights, h_infiniteLights, &kinput.d_lights,
               &kinput.d_infiniteLights);
  kinput.lightCount = h_lights.size();
  kinput.infiniteLightCount = h_infiniteLights.size();
  kinput.d_cam = deviceCamera(h_camera);
  kinput.d_haltonOwen = copyHaltonOwenToDeviceAlloc(blocks, threads);
  kinput.sampleOffset = 0;
  kinput.signalTerm = nullptr;
  CUDA_CHECK(cudaMalloc(&kinput.signalTerm, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(kinput.signalTerm, 0, sizeof(uint32_t)));
}

void freeDeviceMemory(WavefrontInput& kinput) {
  // scene
  cudaFree(kinput.signalTerm);
  cudaFree(kinput.d_haltonOwen);
  cudaFree(kinput.d_triSoup.matId);
  cudaFree(kinput.d_triSoup.xs);
  cudaFree(kinput.d_triSoup.ys);
  cudaFree(kinput.d_triSoup.zs);
  cudaFree(kinput.d_bsdf);
  cudaFree(kinput.d_lights);
  cudaFree(kinput.d_infiniteLights);
  cudaFree(kinput.d_cam);

  // GMEM queues
  decltype(kinput.anyhitQueue)::free(kinput.anyhitQueue);
  decltype(kinput.closesthitQueue)::free(kinput.closesthitQueue);
  decltype(kinput.shadeQueue)::free(kinput.shadeQueue);
  decltype(kinput.missQueue)::free(kinput.missQueue);

  // output buffer
  cudaFree(kinput.d_outBuffer);

  // constant memory
  freeDeviceGGXEnergyPreservingTables();
}

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
      "wave-output-" + std::to_string(data->sample) + ".bmp";
  writeOutputBufferRowMajor(data->hostImage, data->width, data->height,
                            name.c_str());
}

void wavefrontMain() {
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  CUDA_CHECK(cudaFuncSetCacheConfig(wavefrontKernel, cudaFuncCachePreferL1));

  uint32_t threads = 0;
  uint32_t blocks = 0;
  uint32_t sharedBytes = 0;
  optimalBlocksAndThreads(blocks, threads, sharedBytes);
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
  cornellBox(false, &h_scene, &h_lights, &h_infiniteLights, &h_bsdfs,
             &h_camera);

  WavefrontInput kinput{};
  allocateDeviceMemory(threads, blocks, h_scene, h_lights, h_infiniteLights,
                       h_bsdfs, h_camera, kinput);

  // TODO double buffering
  float4* hostImage = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostImage,
                            sizeof(float4) * h_camera.width * h_camera.height));
  auto const callbackData = std::make_unique<CallbackData>();
  callbackData->hostImage = hostImage;
  callbackData->width = h_camera.width;
  callbackData->height = h_camera.height;

  // TODO double buffering
  cudaStream_t streams[2]{};
  cudaEvent_t ready[2]{};
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    CUDA_CHECK(cudaEventCreate(&ready[i]));
  }

  static int constexpr TOTAL_SAMPLES = 2048;
  void* kArgs[] = {&kinput};
  dim3 const gridDim{blocks, 1, 1};
  dim3 const blockDim{threads, 1, 1};
  for (int sOffset = 0; sOffset < TOTAL_SAMPLES; sOffset += h_camera.spp) {
    kinput.sampleOffset += sOffset;
    callbackData->sample = kinput.sampleOffset;
    std::cout << "Launching kernel (" << kinput.sampleOffset << ")"
              << std::endl;
    CUDA_CHECK(cudaLaunchCooperativeKernel(wavefrontKernel, gridDim, blockDim,
                                           kArgs, sharedBytes, streams[0]));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(
        cudaMemcpyAsync(hostImage, kinput.d_outBuffer,
                        sizeof(float4) * h_camera.width * h_camera.height,
                        cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaEventRecord(ready[0], streams[0]));
    CUDA_CHECK(cudaEventSynchronize(ready[0]));
    streamCallbackWriteBuffer(callbackData.get());
  }

  std::cout << "Cleanup..." << std::endl;
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaEventSynchronize(ready[i]));
    CUDA_CHECK(cudaEventDestroy(ready[i]));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
  CUDA_CHECK(cudaFreeHost(hostImage));
  freeDeviceMemory(kinput);
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
