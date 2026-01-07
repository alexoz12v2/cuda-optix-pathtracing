#include "wave-kernels.cuh"

#include "cuda-core/shapes.cuh"

// TODO compare __ldg with .cv pascal PTX Access with normal access
__device__ __forceinline__ HitResult closesthit(
    Ray const& __restrict__ ray, TriangleSoup const& __restrict__ triSoup) {
  HitResult result;
  for (int tri = 0; tri < triSoup.count; ++tri) {
    assert((((uintptr_t)triSoup.xs) & 15) == 0);
    assert((((uintptr_t)triSoup.ys) & 15) == 0);
    assert((((uintptr_t)triSoup.zs) & 15) == 0);
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

__device__ __forceinline__ void handleHit(
    ClosestHitInput const& __restrict__ kinput,
    HitResult const& __restrict__ hitResult,
    DeviceQueue<AnyhitInput>& outAnyHitQueue,
    DeviceQueue<ShadeInput>& outShadeQueue) {
  // 2. Queue Pushes
  unsigned mask = __activemask();
  // TODO SMEM?
  AnyhitInput const anyHitInput{.state = kinput.state,
                                .pos = hitResult.pos,
                                .rayD = kinput.ray.d,
                                .normal = hitResult.normal,
                                .error = hitResult.error,
                                .matId = hitResult.matId,
                                .t = hitResult.t};
  mask = outAnyHitQueue.queuePush(&anyHitInput);
  assert(mask == __activemask());
  ShadeInput const shadeInput{
      .state = kinput.state,
      .pos = hitResult.pos,
      .rayD = kinput.ray.d,
      .normal = hitResult.normal,
      .error = hitResult.error,
      .matId = hitResult.matId,
      .t = hitResult.t,
  };
  mask = outShadeQueue.queuePush(&shadeInput);
  assert(mask == __activemask());

  CH_PRINT("CH [%u %u] px: [%d %d] d: %d | hit at: %f %f %f\n", blockIdx.x,
           threadIdx.x, kinput.state->pixelCoordX, kinput.state->pixelCoordY,
           kinput.state->depth, hitResult.pos.x, hitResult.pos.y,
           hitResult.pos.z);
}

__device__ __forceinline__ void handleMiss(
    ClosestHitInput const& kinput, DeviceQueue<MissInput>& outMissQueue) {
  // if miss -> enqueue miss queue
  MissInput const missInput{.state = kinput.state,
                            .rayDirection = kinput.ray.d};
  unsigned const mask = outMissQueue.queuePush(&missInput);
  assert(mask == __activemask());
}

__global__ void closesthitKernel(DeviceQueue<ClosestHitInput> inQueue,
                                 DeviceQueue<MissInput> outMissQueue,
                                 DeviceQueue<AnyhitInput> outAnyhitQueue,
                                 DeviceQueue<ShadeInput> outShadeQueue,
                                 DeviceHaltonOwen* d_haltonOwen,
                                 TriangleSoup d_triSoup) {
  ClosestHitInput kinput{};  // TODO SMEM?
  DeviceHaltonOwen& warpRng = d_haltonOwen[globalWarpId()];

  int lane = getLaneId();
  int mask = inQueue.queuePop(&kinput);
  while (mask & (1 << lane)) {
    int const activeWorkers = __activemask();
    warpRng.startPixelSample(
        CMEM_haltonOwenParams,
        make_int2(kinput.state->pixelCoordX, kinput.state->pixelCoordY),
        kinput.state->sampleIndex);

    if (auto const hitResult = closesthit(kinput.ray, d_triSoup);
        hitResult.hit) {
      handleHit(kinput, hitResult, outAnyhitQueue, outShadeQueue);
    } else {
      handleMiss(kinput, outMissQueue);
    }

    __syncwarp(activeWorkers);  // TODO is this necessary?
    lane = getCoalescedLaneId(activeWorkers);
    mask = inQueue.queuePop(&kinput);
  }
}
