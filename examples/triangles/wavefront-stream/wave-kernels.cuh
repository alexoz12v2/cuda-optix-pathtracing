#ifndef DMT_WAVEFRONT_STREAM_WAVE_KERNELS_CUH
#define DMT_WAVEFRONT_STREAM_WAVE_KERNELS_CUH

#define DMT_WAVEFRONT_STREAMS  // for queue.cuh

#include "cuda-core/bsdf.cuh"
#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/extra_math.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/queue.cuh"

// #define RG_PRINT(...) printf(__VA_ARGS__)
// #define CH_PRINT(...) printf(__VA_ARGS__)
// #define AH_PRINT(...) printf(__VA_ARGS__)
// #define MS_PRINT(...) printf(__VA_ARGS__)
// #define SH_PRINT(...) printf(__VA_ARGS__)
// #define UTILS_PRINT(...) printf(__VA_ARGS__)

#define RG_PRINT(...)
#define CH_PRINT(...)
#define AH_PRINT(...)
#define MS_PRINT(...)
#define SH_PRINT(...)
#define UTILS_PRINT(...)

struct HostTriangleScene;

// ----------------------------------------------------------------------------
// Path State
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// constants
// ----------------------------------------------------------------------------
extern __constant__ float CMEM_cameraFromRaster[32];
extern __constant__ float CMEM_renderFromCamera[32];
extern __constant__ DeviceHaltonOwenParams CMEM_haltonOwenParams;
extern __constant__ int2 CMEM_imageResolution;
extern __constant__ int2 CMEM_tileResolution;
extern __constant__ int CMEM_spp;

__host__ void allocateDeviceConstantMemory(DeviceCamera const& h_camera,
                                           int xTile, int yTile);
__host__ void freeDeviceConstantMemory();

inline __host__ __device__ __forceinline__ Transform const* arrayAsTransform(
    float const* arr) {
  static_assert(sizeof(Transform) == 32 * sizeof(float) &&
                alignof(Transform) <= 16);
  // 16-byte aligned and at least 32 elements
  return reinterpret_cast<Transform const*>(arr);
}

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------
// TODO AOSOA on warpSize + count?
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

// ----------------------------------------------------------------------------
// device utils
// ----------------------------------------------------------------------------
__device__ __forceinline__ unsigned globalWarpId() {
  return (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
}

struct WavefrontStreamInput {
  WavefrontStreamInput(uint32_t threads, uint32_t blocks,
                       HostTriangleScene const& h_scene,
                       std::vector<Light> const& h_lights,
                       std::vector<Light> const& h_infiniteLights,
                       std::vector<BSDF> const& h_bsdfs,
                       DeviceCamera const& h_camera);
  WavefrontStreamInput(WavefrontStreamInput const&) = delete;
  WavefrontStreamInput(WavefrontStreamInput&&) noexcept = delete;
  WavefrontStreamInput& operator=(WavefrontStreamInput const&) = delete;
  WavefrontStreamInput& operator=(WavefrontStreamInput&&) noexcept = delete;
  ~WavefrontStreamInput() noexcept;

  DeviceQueue<ClosestHitInput> closesthitQueue;
  DeviceQueue<MissInput> missQueue;
  DeviceQueue<AnyhitInput> anyhitQueue;
  DeviceQueue<ShadeInput> shadeQueue;
  DeviceArena<PathState> pathStateSlots;
  DeviceHaltonOwen* d_haltonOwen;
  DeviceCamera* d_cam;
  TriangleSoup d_triSoup;
  Light* d_lights;
  Light* infiniteLights;
  BSDF* d_bsdfs;
  float4* d_outBuffer;
  int sampleOffset;
  uint32_t lightCount;
  uint32_t infiniteLightCount;
};

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

// len(d_haltonOwen) = warp count
__global__ void raygenKernel(DeviceQueue<ClosestHitInput> outQueue,
                             DeviceArena<PathState> pathStateSlots,
                             DeviceHaltonOwen* d_haltonOwen,
                             DeviceCamera* d_cam, int tileIdxX, int tileIdxY,
                             int tileDimX, int tileDimY, int sampleOffset);
__global__ void closesthitKernel(DeviceQueue<ClosestHitInput> inQueue,
                                 DeviceQueue<MissInput> outMissQueue,
                                 DeviceQueue<AnyhitInput> outAnyhitQueue,
                                 DeviceQueue<ShadeInput> outShadeQueue,
                                 TriangleSoup d_triSoup);
__global__ void anyhitKernel(DeviceQueue<AnyhitInput> inQueue, Light* d_lights,
                             uint32_t lightCount, BSDF* d_bsdfs,
                             TriangleSoup d_triSoup);
__global__ void missKernel(DeviceQueue<MissInput> inQueue,
                           DeviceArena<PathState> pathStateSlots,
                           float4* d_outBuffer, Light* infiniteLights,
                           uint32_t infiniteLightCount);
__global__ void shadeKernel(DeviceQueue<ShadeInput> inQueue,
                            DeviceQueue<ClosestHitInput> outQueue,
                            DeviceArena<PathState> pathStateSlots,
                            float4* d_outBuffer, BSDF* d_bsdfs);
__global__ void checkDoneDepth(DeviceArena<PathState> pathStateSlots,
                               int* d_done);

#endif  // DMT_WAVEFRONT_STREAM_WAVE_KERNELS_CUH