#include "wave-kernels.cuh"

#include "cuda-core/shapes.cuh"

// anonymous namespace for static/internal linkage (applies to __device__ too)
namespace {
// TODO compare __ldg with .cv pascal PTX Access with normal access
__device__ __forceinline__ HitResult
closesthit(Ray const& __restrict__ ray,
           TriangleSoup const& __restrict__ triSoup, int transmissionCount) {
  HitResult result;
  for (int tri = 0; tri < triSoup.count; ++tri) {
#if DMT_ENABLE_ASSERTS
    assert((((uintptr_t)triSoup.xs) & 15) == 0);
    assert((((uintptr_t)triSoup.ys) & 15) == 0);
    assert((((uintptr_t)triSoup.zs) & 15) == 0);
#endif
    float4 const x = __ldg(reinterpret_cast<float4 const*>(triSoup.xs) + tri);
    float4 const y = __ldg(reinterpret_cast<float4 const*>(triSoup.ys) + tri);
    float4 const z = __ldg(reinterpret_cast<float4 const*>(triSoup.zs) + tri);
    HitResult const other = triangleIntersect(x, y, z, ray);
    if (other.hit && other.t < result.t) {
      result = other;
      // TODO better? Coalescing?
      result.matId = triSoup.matId[tri];
      // _Important_ Our convention: Geometric and Shading Normals should be
      // directed accordingly to input (outgoing) direction
#if 0
      result.normal = flipIfOdd(result.normal, transmissionCount);
#else
      if (dot(ray.d, result.normal) > 0) {
        // inside surface, flip normal
        result.normal *= -1;
      }
#endif
    }
  }
  return result;
}

__device__ __forceinline__ void handleHit(
    ClosestHitInput const& __restrict__ kinput,
    HitResult const& __restrict__ hitResult,
    QueueType<AnyhitInput>& outAnyHitQueue,
    QueueType<ShadeInput>& outShadeQueue) {
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
#if USE_SIMPLE_QUEUE
  bool pushed = outAnyHitQueue.queuePush<false>(anyHitInput);
#  ifdef DMT_DEBUG
  assert(pushed);
#  else
  if (!pushed) {
    int const deadWorkers = __activemask();
    asm volatile("trap;");
  }
#  endif
#else
  int const coalescedLane = getCoalescedLaneId(__activemask());
  mask = outAnyHitQueue.queuePush<false>(&anyHitInput);
#  ifdef DMT_DEBUG
  assert(1u << coalescedLane & mask);
#  else
  if (!(1u << coalescedLane & mask)) {
    asm volatile("trap;");
  }
#  endif
#endif
  ShadeInput const shadeInput{
      .state = kinput.state,
      .pos = hitResult.pos,
      .rayD = kinput.ray.d,
      .normal = hitResult.normal,
      .error = hitResult.error,
      .matId = hitResult.matId,
      .t = hitResult.t,
  };
#if USE_SIMPLE_QUEUE
  pushed = outShadeQueue.queuePush<false>(shadeInput);
#  ifdef DMT_DEBUG
  assert(pushed);
#  else
  if (!pushed) {
    int const deadWorkers = __activemask();
    asm volatile("trap;");
  }
#  endif
#else
  mask = outShadeQueue.queuePush<false>(&shadeInput);
#  ifdef DMT_DEBUG
  assert(1u << coalescedLane & mask);
#  else
  if (!(1u << coalescedLane & mask)) {
    asm volatile("trap;");
  }
#  endif
#endif

  CH_PRINT("CH [%u %u] px: [%d %d] d: %d | hit at: %f %f %f\n", blockIdx.x,
           threadIdx.x, kinput.state->pixelCoordX, kinput.state->pixelCoordY,
           kinput.state->depth, hitResult.pos.x, hitResult.pos.y,
           hitResult.pos.z);
}

__device__ __forceinline__ void handleMiss(ClosestHitInput const& kinput,
                                           QueueType<MissInput>& outMissQueue) {
  // if miss -> enqueue miss queue
  MissInput const missInput{.state = kinput.state,
                            .rayDirection = kinput.ray.d};
#if USE_SIMPLE_QUEUE
  bool const pushed = outMissQueue.queuePush<false>(missInput);
#  ifdef DMT_DEBUG
  assert(pushed);
#  else
  if (!pushed) {
    asm volatile("trap;");
  }
#  endif
#else
  int const coalescedLane = getCoalescedLaneId(__activemask());
  unsigned const mask = outMissQueue.queuePush<false>(&missInput);
#  ifdef DMT_DEBUG
  assert(1u << coalescedLane & mask);
#  else
  if (!(1u << coalescedLane & mask)) {
    asm volatile("trap;");
  }
#  endif
#endif
}

}  // namespace

__global__ void closesthitKernel(QueueType<ClosestHitInput> inQueue,
                                 QueueType<MissInput> outMissQueue,
                                 QueueType<AnyhitInput> outAnyhitQueue,
                                 QueueType<ShadeInput> outShadeQueue,
                                 TriangleSoup d_triSoup) {
  ClosestHitInput kinput{};  // TODO SMEM?

#if USE_SIMPLE_QUEUE
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < inQueue.queueSize(); idx += blockDim.x * gridDim.x) {
    if (!inQueue.marked[idx]) {
      continue;
    }
    inQueue.marked[idx] = 0;
    kinput = inQueue.buffer[idx];
#else
  int lane = getLaneId();
  int mask = inQueue.queuePop<false>(&kinput);
  while (mask & (1 << lane)) {
#endif
    int const activeWorkers = __activemask();

    // TODO check GMEM accesses here
    if (auto const hitResult = closesthit(
            kinput.ray, d_triSoup, __ldg(&kinput.state->transmissionCount));
        hitResult.hit) {
      handleHit(kinput, hitResult, outAnyhitQueue, outShadeQueue);
    } else {
      handleMiss(kinput, outMissQueue);
    }

    // we want a coalesced pop, hence join hit lanes and miss lanes
    __syncwarp(activeWorkers);
#if USE_SIMPLE_QUEUE
#else
    lane = getCoalescedLaneId(activeWorkers);
    mask = inQueue.queuePop<false>(&kinput);
#endif
  }
}
