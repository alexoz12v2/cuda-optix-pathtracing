#include "wave-kernels.cuh"

#include "cuda-core/host_utils.cuh"
#include "cuda-core/host_scene.cuh"

// TODO double buffering

WavefrontStreamInput::WavefrontStreamInput(
    uint32_t threads, uint32_t blocks, HostTriangleScene const& h_scene,
    std::vector<Light> const& h_lights,
    std::vector<Light> const& h_infiniteLights,
    std::vector<BSDF> const& h_bsdfs, DeviceCamera const& h_camera) {
  // output buffer
  d_outBuffer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_outBuffer,
                        h_camera.width * h_camera.height * sizeof(float4)));
  CUDA_CHECK(cudaMemset(d_outBuffer, 0,
                        h_camera.width * h_camera.height * sizeof(float4)));
  // GMEM queues
  static int constexpr QUEUE_CAP = 1 << 12;
#if USE_SIMPLE_QUEUE
  allocSimpleQueue(anyhitQueue, QUEUE_CAP);
  allocSimpleQueue(closesthitQueue, QUEUE_CAP);
  allocSimpleQueue(shadeQueue, QUEUE_CAP);
  allocSimpleQueue(missQueue, QUEUE_CAP);
#else
  initQueue(anyhitQueue, QUEUE_CAP);
  initQueue(closesthitQueue, QUEUE_CAP << 2);
  initQueue(shadeQueue, QUEUE_CAP);
  initQueue(missQueue, QUEUE_CAP);
#endif

  // scene
  d_triSoup = triSoupFromTriangles(h_scene, h_bsdfs.size());
  d_bsdfs = deviceBSDF(h_bsdfs);
  deviceLights(h_lights, h_infiniteLights, &d_lights, &infiniteLights);
  lightCount = h_lights.size();
  infiniteLightCount = h_infiniteLights.size();
  d_cam = deviceCamera(h_camera);
  d_haltonOwen = copyHaltonOwenToDeviceAlloc(blocks, threads);
  sampleOffset = 0;

  // path states
  initDeviceArena(pathStateSlots, QUEUE_CAP << 8);
}

WavefrontStreamInput::~WavefrontStreamInput() noexcept {
  // path states
  freeDeviceArena(pathStateSlots);

  // scene
  cudaFree(d_haltonOwen);
  cudaFree(d_triSoup.matId);
  cudaFree(d_triSoup.xs);
  cudaFree(d_triSoup.ys);
  cudaFree(d_triSoup.zs);
  cudaFree(d_bsdfs);
  cudaFree(d_lights);
  cudaFree(infiniteLights);
  cudaFree(d_cam);

  // GMEM queues
#if USE_SIMPLE_QUEUE
  freeSimpleQueue(anyhitQueue);
  freeSimpleQueue(closesthitQueue);
  freeSimpleQueue(shadeQueue);
  freeSimpleQueue(missQueue);
#else
  freeQueue(anyhitQueue);
  freeQueue(closesthitQueue);
  freeQueue(shadeQueue);
  freeQueue(missQueue);
#endif

  // output buffer
  cudaFree(d_outBuffer);
}

#if USE_SIMPLE_QUEUE
static int constexpr LOG_NUM_BANKS = 5;
constexpr auto CONFLICT_FREE_OFFSET = [](int n) {
  return n >> LOG_NUM_BANKS + n >> (2 * LOG_NUM_BANKS);
};

// Given:
// - buffer: [ A, -, B, C, -, -, D ]
// - marked: [ 1, 0, 1, 1, 0, 0, 1 ]
// - count  = 7
// After compaction:
// - buffer: [ A, B, C, D, ?, ?, ? ]
// - marked: [ 1, 1, 1, 1, 0, 0, 0 ]
// - count  = 4
// Key idea (assume 1D grid, 1D block, _called with full block_)
// - Each block scans a chunk ( 4 Bytes per tid of dynamic SMEM at offset 0)
// - Compute local offsets via prefix sum
// - Write compacted output
// - One atomic per block
template <typename T>
__device__ void compact(SimpleDeviceQueue<T>& q) {
  // instead of computing a global prefix sum to know the positions of each
  // object in stream compaction, do a prefix sum for each active warp and
  // proceed sequentially
  // - Pros: 1) Simpler 2) no SMEM required 3) No multiple kernel invocations
  // - note: Still requires intervention fron host side to swap buffers

  // count is left untouched by all blocks
  // Note: count here is stale information, as each thread which processes the
  // values put marked[i] = 0 without touching count. it is to be interpreted
  // as the maximum values pushed in this iteration
  int const elementCount = q.queueSize();

  // Warning: global count. Should have been already reset
  // backCount = 0!!

  // grid-stride loop over existing buffer
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elementCount;
       i += blockDim.x * gridDim.x) {
    int const isAlive = q.marked[i];
    unsigned const activeMask = __activemask();

    // warp-level prefix sum
    int const laneId = getLaneId();
    // this is the "take the max" part of the warp by warp prefix sum
    int const warpSum = __popc(__ballot_sync(activeMask, isAlive));
    if (isAlive) {
      // active pushers rank (first see who pushes, then filter superior lanes,
      // finally pop count)
      int const aliveMask = __activemask();
      int const rank =
          __popc(__ballot_sync(aliveMask, isAlive) & ((1 << laneId) - 1));
      int warpBase = -1;
      // warp coalesced leader (ffs - 1 satisfying ballot condition) reserves
      // space in back buffer
      if (rank == 0) {
        warpBase = atomicAdd(q.backCount, warpSum);
      }
      warpBase = __shfl_sync(aliveMask, warpBase,
                             __ffs(__ballot_sync(aliveMask, isAlive)) - 1);
      // write
      q.backBuffer[warpBase + rank] = q.buffer[i];
      q.backMarked[warpBase + rank] = 1;
    }
  }

  // counter management. (swap count <- backCount and backCount to 0:  host)
}
#endif

__global__ void checkDoneDepth(DeviceArena<PathState> pathStateSlots,
                               QueueType<ClosestHitInput> closesthitQueue,
                               QueueType<MissInput> missQueue,
                               QueueType<AnyhitInput> anyhitQueue,
                               QueueType<ShadeInput> shadeQueue, int* d_done) {
#if USE_SIMPLE_QUEUE
  int res = 0;
  if (getWarpId() == 0) {
    res = pathStateSlots.empty_agg();
  }
  res = __shfl_sync(0xFFFF'FFFFU, res, 0);

  compact(closesthitQueue);
  compact(missQueue);
  compact(anyhitQueue);
  compact(shadeQueue);

  if (res) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *d_done = 0;
      if (closesthitQueue.queueSize() == 0 && missQueue.queueSize() == 0 &&
          anyhitQueue.queueSize() == 0 && shadeQueue.queueSize() == 0) {
#  ifdef DMT_DEBUG
        UTILS_PRINT("UTILS: done depth | %d\n", res);
#  endif
        *d_done = 1;
      }
    }
  }

#else
  if (blockIdx.x == 0 && getWarpId() == 0) {
    int const res = pathStateSlots.empty_agg();
    if (threadIdx.x == 0) {
      UTILS_PRINT("UTILS: done depth | %d\n", res);
      *d_done = res;
    }
  }
#endif
}
