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

#define WAVEFRONT_KERNEL_RAYGEN_ACTIVE 1
#define WAVEFRONT_KERNEL_CLOSESTHIT_ACTIVE 1
#define WAVEFRONT_KERNEL_ANYHIT_ACTIVE 1
#define WAVEFRONT_KERNEL_MISS_ACTIVE 1
#define WAVEFRONT_KERNEL_SHADE_ACTIVE 1

#define RG_PRINT(...) printf(__VA_ARGS__)
// #define CH_PRINT(...) printf(__VA_ARGS__)
// #define AH_PRINT(...) printf(__VA_ARGS__)
// #define MS_PRINT(...) printf(__VA_ARGS__)
// #define SH_PRINT(...) printf(__VA_ARGS__)

// #define RG_PRINT(...)
#define CH_PRINT(...)
#define AH_PRINT(...)
#define MS_PRINT(...)
#define SH_PRINT(...)

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
  __device__ static void make(PathState& state, int px, int py, int s, int spp,
                              int slot) {
    state.pixelCoordX = px;
    state.pixelCoordY = py;
    state.sampleIndex = s;
    state.spp = spp;
    state.depth = 0;
    state.throughput = make_float3(1, 1, 1);
    state.L = make_float3(0, 0, 0);
    state.lastBsdfPdf = 0;
    state.lastBounceTransmission = 0;
    state.anySpecularBounces = 0;
    state.useCount = 0;
    state.bufferSlot = slot;
  }

  void __forceinline__ __device__ ldg_pxs(int& px, int& py, int& s) {
    px = __ldg(&pixelCoordX);
    py = __ldg(&pixelCoordY);
    s = __ldg(&sampleIndex);
  }

  void __forceinline__ __device__ ldg_px(int& px, int& py) {
    px = __ldg(&pixelCoordX);
    py = __ldg(&pixelCoordY);
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
  int lastBounceTransmission;  // TODO cache in SMEM?
  int anySpecularBounces;
  int useCount;
  int bufferSlot;
};
static_assert(sizeof(PathState) % alignof(PathState) == 0);

__device__ __forceinline__ void freeState(
    DeviceArena<PathState>& pathStateSlots, PathState* state) {
  if (state) {
    pathStateSlots.free_slot(state->bufferSlot);
#ifdef DMT_DEBUG
    memset(state, 0, sizeof(PathState));
    __threadfence();
#endif
  }
}

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
  DeviceQueue<ClosestHitInput> closesthitQueue;
  DeviceQueue<AnyhitInput> anyhitQueue;
  DeviceQueue<MissInput> missQueue;
  DeviceQueue<ShadeInput> shadeQueue;

  // help me
  DeviceArena<PathState> pathStateSlots;
};
// kernel param size limit (cc < 7.0)
static_assert(sizeof(WavefrontInput) <= 4096 * 1024);

inline __device__ __forceinline__ void raygen(
    RaygenInput const& raygenInput, DeviceArena<PathState>& pathStateSlots,
    DeviceHaltonOwen& warpRng, DeviceHaltonOwenParams const& params,
    ClosestHitInput& out) {
  int2 const pixel = make_int2(raygenInput.px, raygenInput.py);
  CameraSample const cs = getCameraSample(pixel, warpRng, params);
  // TODO aligned allocation
  static unsigned constexpr MAX_RETRY = 16;
  static unsigned constexpr EXP_BACKOFF_SHFT = 1;
  static unsigned constexpr WAIT_BASE = 3600;
  static unsigned constexpr WAIT_MAX = 0xFFFF'FFFFU;
  static_assert(WAIT_BASE * (MAX_RETRY | 1) << EXP_BACKOFF_SHFT != 0U);
  int attempt = 0;
  int wait = WAIT_BASE;
  assert(!out.state);
  int slot = -1;
  while (slot < 0) {
    slot = pathStateSlots.allocate();
    pascal_fixed_sleep(64);
#if PRINT_ASSERT
    if (slot < 0) {
      if (auto const g = cg::coalesced_threads(); g.thread_rank() == 0) {
        RG_PRINT(
            "RG [%u %u] (cg: base: %u count %u) slot allocation failed for px "
            "%d %d s %d\n",
            blockIdx.x, threadIdx.x, getLaneId(), g.size(), raygenInput.px,
            raygenInput.py, raygenInput.sampleIndex);
      }
    }
#endif
    // asm volatile("trap;");
  }
  out.state = &pathStateSlots.buffer[slot];

  PathState::make(*out.state, raygenInput.px, raygenInput.py,
                  raygenInput.sampleIndex, raygenInput.spp, slot);
  __threadfence();
  out.ray = getCameraRay(cs, *raygenInput.cameraFromRaster,
                         *raygenInput.renderFromCamera);
}

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

inline constexpr int WAVEFRONT_KERNEL_SHARED_MEM_BYTES = 12;

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

  DeviceHaltonOwen& warpRng =
      input.d_haltonOwen[(blockIdx.x * blockDim.x + threadIdx.x) / warpSize];

  if (blockType == WAVEFRONT_KERNEL_RAYGEN_DIVISOR) {
#if WAVEFRONT_KERNEL_RAYGEN_ACTIVE
    static int constexpr px = 8;
    static int constexpr py = 4;
    int& warpRemaining = SMEM[0];

    // 1. Determine our position in the subdivided grid
    // we assume only the (gridDim / WAVEFRONT_KERNEL_TYPES) handle raygen
    int const raygenGridDim = gridDim.x / WAVEFRONT_KERNEL_TYPES;

    // 2. Calculate warp ID and Stride (TODO check PTX)
    assert(warpSize == 32);
    int const threadsPerBlock = blockDim.x;
    int const laneId = threadIdx.x % 32;
    int const warpIdInBlock = threadIdx.x / 32;
    int const warpsPerBlock = threadsPerBlock / 32;
    if (threadIdx.x == 0) {
      warpRemaining = warpsPerBlock;
    }
    __syncthreads();  // bar.sync 0;

    // Global warp ID across the raygen portion of the grid
    int const globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    int const totalWarps = raygenGridDim * warpsPerBlock;

    // 3. Define Work Space
    int const numTilesX = ceilDiv(CMEM_imageResolution.x, px);
    int const numTilesY = ceilDiv(CMEM_imageResolution.y, py);
    int const totalTiles = numTilesX * numTilesY;
    int const totalWorkUnits = totalTiles * CMEM_spp;

    // 4. Grid-Stride loop: each warp grabs a work unit (8x4 tile for _1_
    // specific sample)
    for (int i = globalWarpId; i < totalWorkUnits; i += totalWarps) {
      // Decode 1D index into (Tile, Sample)
      assert(i / totalTiles < CMEM_spp);
      int const sampleIdx = i / totalTiles + input.sampleOffset;
      int const tileIdx = i % totalTiles;

      // Decode tile index into 2D coordinates
      int const tileY = tileIdx / numTilesX;
      int const tileX = tileIdx % numTilesX;

      // 5. Map threads within the warp into pixels in the 8x4 tile (row major)
      int const x = tileX * px + (laneId % px);
      int const y = tileY * py + (laneId / px);

      // boundary check for image res not divisible by 8x4
      if (x < CMEM_imageResolution.x && y < CMEM_imageResolution.y) {
        // generate ray for (x, y) at sampleIndex
        {
          // construct raygen input
          ClosestHitInput raygenElem{};  // TODO SMEM?
          {
            // TODO without struct
            RaygenInput const raygenInput = {
                .px = x,
                .py = y,
                .sampleIndex = sampleIdx,
                .spp = input.d_cam->spp,
                .cameraFromRaster = arrayAsTransform(CMEM_cameraFromRaster),
                .renderFromCamera = arrayAsTransform(CMEM_renderFromCamera),
            };
            warpRng.startPixelSample(CMEM_haltonOwenParams, make_int2(x, y),
                                     sampleIdx);
            raygen(raygenInput, input.pathStateSlots, warpRng,
                   CMEM_haltonOwenParams, raygenElem);
            assert(raygenElem.state);
          }
          // check if 32 available places in queue
          unsigned active = __activemask();
          while (active) {
            // warp-wide push (push already does __threadfence())
            active &= ~input.closesthitQueue.queuePush(&raygenElem);
            if (active) {
              pascal_fixed_sleep(64);
              RG_PRINT(
                  "RG [%u %u] RAYGEN push failed for sample [%d %d] %d. head: "
                  "%d tail: %d\n",
                  blockIdx.x, threadIdx.x, x, y, sampleIdx,
                  *(volatile int*)(&input.closesthitQueue.head),
                  *(volatile int*)(&input.closesthitQueue.tail));
            }
          }
          RG_PRINT(
              "RG [%u %u] RAYGEN activemask 0x%x generated sample [%d %d] %d\n",
              blockIdx.x, threadIdx.x, __activemask(), x, y, sampleIdx);
        }
      }
    }
    __syncwarp();

    RG_PRINT("RG [%u %u] RAYGEN activemask 0x%x finished generation\n",
             blockIdx.x, threadIdx.x, __activemask());

    // warp leader decrements warpRemaining
    if ((threadIdx.x & 31) == 0) {
      // if warpRemaining zero (not negative),
      // warp leader increments GMEM signalTerm
      if (atomicSub(&warpRemaining, 1) == 1) {
#  if 1
        if (atomicInc(input.signalTerm, 1) == blockNumPerKernel - 1) {
          printf("[%u] RG GLOBALLY FINISHED RAYGEN\n", blockIdx.x);
        }
#  else
        atomicInc(input.signalTerm, 1);
#  endif
      }
    }
#endif
  } else if (blockType == WAVEFRONT_KERNEL_CLOSESTHIT_DIVISOR) {
#if WAVEFRONT_KERNEL_CLOSESTHIT_ACTIVE
    // consumer preamble
    bool extraLife = true;
    while (true) {
      // used for any vote
      bool finished = false;
      assert(__activemask() == 0xFFFF'FFFFU);

      ClosestHitInput kinput{};  // TODO SMEM?
      if (int const mask = input.closesthitQueue.queuePop(&kinput);
          mask & (1 << getLaneId())) {
        // TODO optimize GMEM access
        warpRng.startPixelSample(
            CMEM_haltonOwenParams,
            make_int2(kinput.state->pixelCoordX, kinput.state->pixelCoordY),
            kinput.state->sampleIndex);
        if (auto const hitResult = closesthit(kinput.ray, input.d_triSoup);
            hitResult.hit) {
          extraLife = true;
          assert(kinput.state);
          // if hit -> enqueue anyhit queue and shade queue

          // 1. Set useCount to 1 implies "One AnyHit worker is active/pending"
          // We do this BEFORE pushing to queues to prevent Shade from freeing
          // early.
          kinput.state->useCount = 1;
          __threadfence();  // Ensure the write is visible before the push makes
                            // the pointer visible

          // 2. Queue Pushes
          int const hitWorkersMask = __activemask();
          int const coalescedLaneId = getCoalescedLaneId(hitWorkersMask);
          // TODO SMEM?
          {
            // scope for struct
            int coalescedMask = 0;
            AnyhitInput const anyHitInput{.state = kinput.state,
                                          .pos = hitResult.pos,
                                          .rayD = kinput.ray.d,
                                          .normal = hitResult.normal,
                                          .error = hitResult.error,
                                          .matId = hitResult.matId,
                                          .t = hitResult.t};
            // Warning: Potential Deadlock
            do {
              coalescedMask = input.anyhitQueue.queuePush(&anyHitInput);
              if (!(coalescedMask & (1 << coalescedLaneId))) {
                pascal_fixed_sleep(64);
              }
            } while (!(coalescedMask & (1 << coalescedLaneId)));
          }
          // TODO SMEM?
          {
            int coalescedMask = 0;
            // scope for struct
            ShadeInput const shadeInput{
                .state = kinput.state,
                .pos = hitResult.pos,
                .rayD = kinput.ray.d,
                .normal = hitResult.normal,
                .error = hitResult.error,
                .matId = hitResult.matId,
                .t = hitResult.t,
            };
            // Warning: Potential Deadlock
            do {
              coalescedMask = input.shadeQueue.queuePush(&shadeInput);
              if (!(coalescedMask & (1 << coalescedLaneId))) {
                pascal_fixed_sleep(64);
              }
            } while (!(coalescedMask & (1 << coalescedLaneId)));
          }

          CH_PRINT("CH [%u %u] px: [%d %d] d: %d | hit at: %f %f %f\n",
                   blockIdx.x, threadIdx.x, kinput.state->pixelCoordX,
                   kinput.state->pixelCoordY, kinput.state->depth,
                   hitResult.pos.x, hitResult.pos.y, hitResult.pos.z);
        } else {
          int const missWorkersMask = __activemask();
          int const coalescedLaneId = getCoalescedLaneId(missWorkersMask);
          // if miss -> enqueue miss queue
          MissInput const missInput{.state = kinput.state,
                                    .rayDirection = kinput.ray.d};
          int coalescedMask = 0;
          do {
            coalescedMask = input.missQueue.queuePush(&missInput);
            if (!(coalescedMask & (1 << coalescedLaneId))) {
              pascal_fixed_sleep(64);
            }
          } while (!(coalescedMask & (1 << coalescedLaneId)));
        }
      } else {
        int const sleepingWorkersMask = __activemask();
        // if input queue empty and raygen done
        bool const shadeEmpty = input.shadeQueue.empty_agg();
        if (int const leader = __ffs(sleepingWorkersMask) - 1;
            shadeEmpty && leader == getLaneId()) {
          // compute if there's no more work to do
          finished = (*input.signalTerm) == blockNumPerKernel && !extraLife;
          extraLife = false;
#  if 1
          if (finished) {
            printf("CH [%u %u] ClosestHit Finished\n", blockIdx.x, threadIdx.x);
          }
#  endif
        }
        // something to do but queue not ready. Busy sleep (Pascal doesn't
        // support __nanosleep)
        pascal_fixed_sleep(64);
      }

      // did anyone detect finished condition? If so, Die.
      if (__any_sync(0xFFFF'FFFFU, finished)) {
        break;
      }
    }  // end of persistent kernel loop
#endif
  } else if (blockType == WAVEFRONT_KERNEL_ANYHIT_DIVISOR) {
#if WAVEFRONT_KERNEL_ANYHIT_ACTIVE
    bool extraLife = true;
    while (true) {
      // used for any vote
      bool finished = false;
      assert(__activemask() == 0xFFFF'FFFFU);
      AnyhitInput kinput{};  // TODO SMEM?
      if (int const mask = input.anyhitQueue.queuePop(&kinput);
          mask & (1 << getLaneId())) {
        // useCount on state set to one by closesthit
        // configure RNG
        warpRng.startPixelSample(
            CMEM_haltonOwenParams,
            make_int2(kinput.state->pixelCoordX, kinput.state->pixelCoordY),
            kinput.state->sampleIndex);
        // choose a light and prepare BSDF data from intersection
        // TODO maybe: readonly function to fetch 32 bytes with __ldg?
        int const lightIdx =
            min(static_cast<int>(warpRng.get1D(CMEM_haltonOwenParams) *
                                 input.lightCount),
                input.lightCount - 1);
        Light const light = input.d_lights[lightIdx];
        BSDF const bsdf = [&] __device__() {  // TODO copy?
          BSDF theBsdf = input.d_bsdf[kinput.matId];
          prepareBSDF(&theBsdf, kinput.normal, -kinput.rayD);
          return theBsdf;
        }();

        // cast a shadow ray and perform a anyhit query
        // if hit -> NEE
        float3 Le{0, 0, 0};
        float3 const beta = kinput.state->throughput;
        anyhitNEE(-kinput.rayD, beta, kinput.pos, kinput.error, bsdf, light,
                  input.d_triSoup, kinput.normal, input.lightCount,
                  atomicAdd(&kinput.state->lastBounceTransmission, 0),
                  warpRng.get2D(CMEM_haltonOwenParams), &Le);
        atomicAdd(&kinput.state->L.x, Le.x);
        atomicAdd(&kinput.state->L.y, Le.y);
        atomicAdd(&kinput.state->L.z, Le.y);

        // 3. Decrement state useCount when finished
        // We use atomicSub each thread in the warp works on a diffeent item.
        atomicSub(&kinput.state->useCount, 1);

        AH_PRINT(
            "AH [%u %u] px: [%d %d] d: %d | light: %d BSDF: %d | MIS Weighted "
            "Le: "
            "%f %f %f\n",
            blockIdx.x, threadIdx.x, kinput.state->pixelCoordX,
            kinput.state->pixelCoordY, kinput.state->depth, lightIdx,
            kinput.matId, Le.x, Le.y, Le.z);

      } else {
        int const sleepingWorkersMask = __activemask();
        bool closestHitEmpty = input.closesthitQueue.empty_agg();
        if (int const leader = __ffs(sleepingWorkersMask) - 1;
            closestHitEmpty && leader == getLaneId()) {
          // compute if there's no more work to do
          finished = (*input.signalTerm) == blockNumPerKernel && !extraLife;
          extraLife = false;
#  if 1
          if (finished) {
            printf("[%u %u] Anyhit Finished\n", blockIdx.x, threadIdx.x);
          }
#  endif
        }
        // something to do but queue not ready. Busy sleep (Pascal doesn't
        // support __nanosleep)
        pascal_fixed_sleep(64);
      }

      // did anyone detect finished condition? If so, Die.
      if (__any_sync(0xFFFF'FFFFU, finished)) {
        break;
      }
    }  // end persistent kernel thread loop
#endif
  } else if (blockType == WAVEFRONT_KERNEL_MISS_DIVISOR) {
#if WAVEFRONT_KERNEL_MISS_ACTIVE
    // Sink: TODO SMEM array to cache PathStates which are still used?
    // consumer preamble
    bool extraLife = true;

    while (true) {
      // used for any vote
      bool finished = false;
      assert(__activemask() == 0xFFFF'FFFFU);

      MissInput kinput{};
      if (int const mask = input.missQueue.queuePop(&kinput);
          mask & (1 << getLaneId())) {
        // TODO if malloc aligned, use int4
        int const px = __ldg(&kinput.state->pixelCoordX);
        int const py = __ldg(&kinput.state->pixelCoordY);
        int const sampleIndex = __ldg(&kinput.state->sampleIndex);
        // initialize RNG
        warpRng.startPixelSample(CMEM_haltonOwenParams, make_int2(px, py),
                                 sampleIndex);

        // 1. choose an infinite light, fetch last BSDF intersection (TODO)
        int const lightIndex =
            min(static_cast<int>(warpRng.get1D(CMEM_haltonOwenParams) *
                                 input.infiniteLightCount),
                input.infiniteLightCount - 1);
        Light const light = input.d_infiniteLights[lightIndex];
        // 2. add to state radiance
        float pdf = 0;
        float3 Le = evalInfiniteLight(light, kinput.rayDirection, &pdf);
        if (pdf) {
          float3 const beta = kinput.state->throughput;
          Le = beta * Le * input.infiniteLightCount;
          // TODO: If any specular bounce or depth == 0, don't do MIS
          // TODO: Use last BSDF for MIS
          atomicAdd(&kinput.state->L.x, Le.x);
          atomicAdd(&kinput.state->L.y, Le.y);
          atomicAdd(&kinput.state->L.z, Le.z);
        }
        // 2. sinking management: wait for any-hit of current depth
        int volatile* useCount = &kinput.state->useCount;
        while (*useCount > 0) {
          pascal_fixed_sleep(64);
        }
        float4 color = *reinterpret_cast<float4*>(&kinput.state->L);
        color.w = 1;  // TODO filter weight?
        freeState(input.pathStateSlots, kinput.state);
        kinput.state = nullptr;

        // 3. sink to output buffer
        // no need for atomic operation here
        input.d_outBuffer[py * CMEM_imageResolution.x + px] += color;
        MS_PRINT("MS [%u %u] px: [%d %d] | c: %f %f %f\n", blockIdx.x,
                 threadIdx.x, px, py, color.x, color.y, color.z);
      } else {
        int const sleepingWorkersMask = __activemask();
        bool const closestHitEmpty = input.closesthitQueue.empty_agg();
        if (int const leader = __ffs(sleepingWorkersMask) - 1;
            closestHitEmpty && leader == getLaneId()) {
          // compute if there's no more work to do
          finished = (*input.signalTerm) == blockNumPerKernel && !extraLife;
          extraLife = false;
#  if 1
          if (finished) {
            printf("MS [%u %u] Miss Finished\n", blockIdx.x, threadIdx.x);
          }
#  endif
          // something to do but queue not ready. Busy sleep (Pascal doesn't
          // support __nanosleep)
          pascal_fixed_sleep(64);
        }
      }

      // did anyone detect finished condition? If so, Die.
      if (__any_sync(0xFFFF'FFFFU, finished)) {
        break;
      }
    }  // end persistent thread loop
#endif
  } else if (blockType == WAVEFRONT_KERNEL_SHADE_DIVISOR) {
#if WAVEFRONT_KERNEL_SHADE_ACTIVE
    // Sink: TODO SMEM array to cache PathStates which are still used?
    // consumer preamble
    bool extraLife = true;

    while (true) {
      // used for any vote
      bool finished = false;
      assert(__activemask() == 0xFFFF'FFFFU);

      ShadeInput kinput{};
      if (int const mask = input.shadeQueue.queuePop(&kinput);
          mask & (1 << getLaneId())) {
        // TODO if malloc aligned, use int4
        assert(kinput.state);
        int px, py, sampleIndex;
        kinput.state->ldg_pxs(px, py, sampleIndex);
        // initialize RNG
        warpRng.startPixelSample(CMEM_haltonOwenParams, make_int2(px, py),
                                 sampleIndex);

        // Inside each bsdf
        // 1. if max depth reached, kill path
        bool pathDied = false;
        static int constexpr MAX_DEPTH = 3;  // dies afterwards
        // assumes warps have different states
        int const oldDepth = groupedAtomicIncLeaderOnly(&kinput.state->depth);
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
          BSDFSample const bs =
              sampleBsdf(bsdf, -kinput.rayD, kinput.normal, kinput.normal,
                         warpRng.get2D(CMEM_haltonOwenParams),
                         warpRng.get1D(CMEM_haltonOwenParams));
          if (bs) {
            wi = bs.wi;
            // 3. (not dead) update path state
            atomicAdd(&kinput.state->anySpecularBounces, (int)bs.delta);
            atomicExch(&kinput.state->lastBounceTransmission, (int)bs.refract);
            float3 beta = kinput.state->throughput * bs.f *
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
#  define KILL_PATH 0
#  if KILL_PATH
        pathDied = true;
#  endif

        if (pathDied) {
          // 5. if path killed, sink to output
          // no need for atomic operation here

          // sinking management: wait for any-hit of current depth
          int volatile* useCount = &kinput.state->useCount;
          while (*useCount > 0) {
            pascal_fixed_sleep(64);
          }
          float4 color = *reinterpret_cast<float4*>(&kinput.state->L);
          color.w = 1;  // TODO filter weight?
          freeState(input.pathStateSlots, kinput.state);
          kinput.state = nullptr;

          // 3. sink to output buffer
          // no need for atomic operation here
          input.d_outBuffer[py * CMEM_imageResolution.x + px] += color;
          SH_PRINT("SH [%u %u]  px [%u %u] d: %d | path died\n", blockIdx.x,
                   threadIdx.x, px, py, oldDepth);
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
            coalescedMask = input.closesthitQueue.queuePush(&closestHitInput);
            if (!(coalescedMask & (1 << gAlive.thread_rank()))) {
              pascal_fixed_sleep(64);
            }
          } while (!(coalescedMask & (1 << gAlive.thread_rank())));
          SH_PRINT("SH [%u %u]  px [%u %u] d: %d | pushed to closesthit\n",
                   blockIdx.x, threadIdx.x, px, py, oldDepth);
        }
      } else {  // push failed on current lane
        int const sleepingWorkersMask = __activemask();
        bool const closestHitEmpty = input.closesthitQueue.empty_agg();
        if (int const leader = __ffs(sleepingWorkersMask) - 1;
            closestHitEmpty && leader == getLaneId()) {
          // compute if there's no more work to do
          finished = (*input.signalTerm) == blockNumPerKernel && !extraLife;
          extraLife = false;
#  if 1
          if (finished) {
            printf("SH [%u %u] Shade Finished\n", blockIdx.x, threadIdx.x);
          }
#  endif
          // something to do but queue not ready. Busy sleep (Pascal doesn't
          // support __nanosleep)
          pascal_fixed_sleep(64);
        }
      }

      // did anyone detect finished condition? If so, Die.
      if (__any_sync(0xFFFF'FFFFU, finished)) {
        break;
      }
    }  // end persistent kernel thread loop
#endif
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
  // RG: 4 bytes (warpRemaining)
  // sinks: 256 bytes (16 x 8 bytes (addresses) = 1 address per 2 banks
  sharedBytes = WAVEFRONT_KERNEL_SHARED_MEM_BYTES;

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
  initQueue(kinput.anyhitQueue, QUEUE_CAP);
  initQueue(kinput.closesthitQueue, QUEUE_CAP);
  initQueue(kinput.shadeQueue, QUEUE_CAP);
  initQueue(kinput.missQueue, QUEUE_CAP);

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

  // path states
  initDeviceArena(kinput.pathStateSlots, 2048);
}

void freeDeviceMemory(WavefrontInput& kinput) {
  // path states
  freeDeviceArena(kinput.pathStateSlots);

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
  freeQueue(kinput.anyhitQueue);
  freeQueue(kinput.closesthitQueue);
  freeQueue(kinput.shadeQueue);
  freeQueue(kinput.missQueue);

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
  cornellBox(true, &h_scene, &h_lights, &h_infiniteLights, &h_bsdfs, &h_camera);

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

#define ONLY_ONE_EXEC 1

  // TODO small optimization with cuda graphs?

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

#define INSPECT_CLOSEST_HIT 0
#if INSPECT_CLOSEST_HIT
    auto const theInput = std::make_unique<ClosestHitInput[]>(
        kinput.closesthitQueue.queueCapacity);
    CUDA_CHECK(cudaMemcpy(
        theInput.get(), kinput.closesthitQueue.queue,
        kinput.closesthitQueue.queueCapacity * sizeof(ClosestHitInput),
        cudaMemcpyDeviceToHost));
    printf("Some stuff\n");
#endif

#if ONLY_ONE_EXEC
    break;
#endif
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
