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
//#define UTILS_PRINT(...) printf(__VA_ARGS__)

#define RG_PRINT(...)
#define CH_PRINT(...)
#define AH_PRINT(...)
#define MS_PRINT(...)
#define SH_PRINT(...)
 #define UTILS_PRINT(...)

struct HostTriangleScene;

// ----------------------------------------------------------------------------
// Compile time config
// ----------------------------------------------------------------------------
#define USE_SIMPLE_QUEUE 1
#if USE_SIMPLE_QUEUE
template <typename T>
using QueueType = SimpleDeviceQueue<T>;
#else
template <typename T>
using QueueType = DeviceQueue<T>;
#endif

#define FORCE_ATOMIC_OPS 0

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
    state.transmissionCount = 0;
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
  float3 throughput;
  float3 L;
  float lastBsdfPdf;
  int lastBounceTransmission;  // TODO cache in SMEM?
  int transmissionCount;
  int anySpecularBounces;
  int bufferSlot;
};
static_assert(sizeof(PathState) % alignof(PathState) == 0);

__device__ __forceinline__ void freeState(
    DeviceArena<PathState>& pathStateSlots, PathState* state) {
#if DMT_ENABLE_ASSERTS
  assert(state);
#endif
  if (state) {
    pathStateSlots.free_slot(state->bufferSlot);
    // memset(state, 0, sizeof(PathState));
    // __threadfence();
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

  __host__ void swapBuffersAllQueues(cudaStream_t stream) {
#if USE_SIMPLE_QUEUE
    closesthitQueue.swapBuffers(stream);
    missQueue.swapBuffers(stream);
    anyhitQueue.swapBuffers(stream);
    shadeQueue.swapBuffers(stream);
#endif
  }

  QueueType<ClosestHitInput> closesthitQueue;
  QueueType<MissInput> missQueue;
  QueueType<AnyhitInput> anyhitQueue;
  QueueType<ShadeInput> shadeQueue;
  DeviceArena<PathState> pathStateSlots;
  DeviceHaltonOwen* d_haltonOwen;
  DeviceCamera* d_cam;
  TriangleSoup d_triSoup;
  Light* d_lights;
  Light* infiniteLights;
  BSDF* d_bsdfs;
#if !DMT_ENABLE_MSE
  float4* d_outBuffer;
#endif
  int sampleOffset;
  uint32_t lightCount;
  uint32_t infiniteLightCount;
};

__device__ __forceinline__ float3& fp4to3(float4& f) {
  return *reinterpret_cast<float3*>(&f);
}

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

// len(d_haltonOwen) = warp count
__global__ void raygenKernel(QueueType<ClosestHitInput> outQueue,
                             DeviceArena<PathState> pathStateSlots,
                             DeviceHaltonOwen* d_haltonOwen,
                             DeviceCamera* d_cam, int tileIdxX, int tileIdxY,
                             int tileDimX, int tileDimY, int sampleOffset);
__global__ void closesthitKernel(QueueType<ClosestHitInput> inQueue,
                                 QueueType<MissInput> outMissQueue,
                                 QueueType<AnyhitInput> outAnyhitQueue,
                                 QueueType<ShadeInput> outShadeQueue,
                                 TriangleSoup d_triSoup);
__global__ void anyhitKernel(QueueType<AnyhitInput> inQueue, Light* d_lights,
                             uint32_t lightCount, BSDF* d_bsdfs,
                             TriangleSoup d_triSoup);
__global__ void missKernel(QueueType<MissInput> inQueue,
                           DeviceArena<PathState> pathStateSlots,
#if DMT_ENABLE_MSE
                           DeviceOutputBuffer d_out,
#else
                           float4* d_outBuffer,
#endif
                           Light* infiniteLights, uint32_t infiniteLightCount);
__global__ void shadeKernel(QueueType<ShadeInput> inQueue,
                            QueueType<ClosestHitInput> outQueue,
                            DeviceArena<PathState> pathStateSlots,
#if DMT_ENABLE_MSE
                            DeviceOutputBuffer d_out,
#else
                            float4* d_outBuffer,
#endif
                            BSDF* d_bsdfs);
__global__ void checkDoneDepth(DeviceArena<PathState> pathStateSlots,
                               QueueType<ClosestHitInput> closesthitQueue,
                               QueueType<MissInput> missQueue,
                               QueueType<AnyhitInput> anyhitQueue,
                               QueueType<ShadeInput> shadeQueue, int* d_done);

#endif  // DMT_WAVEFRONT_STREAM_WAVE_KERNELS_CUH
