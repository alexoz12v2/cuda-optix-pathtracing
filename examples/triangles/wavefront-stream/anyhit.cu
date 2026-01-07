#include "wave-kernels.cuh"

#include "cuda-core/shapes.cuh"

__device__ __forceinline__ void anyhitNEE(
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

__global__ void anyhitKernel(DeviceQueue<AnyhitInput> inQueue,
                             DeviceHaltonOwen* d_haltonOwen, Light* d_lights,
                             uint32_t lightCount, BSDF* d_bsdfs,
                             TriangleSoup d_triSoup) {
  DeviceHaltonOwen& warpRng = d_haltonOwen[globalWarpId()];

  AnyhitInput kinput{};  // TODO SMEM?
  int mask = inQueue.queuePop(&kinput);
  int lane = getLaneId();
  while (mask & (1 << lane)) {
    int const activeWorkers = __activemask();

    // useCount on state set to one by closesthit
    // configure RNG
    warpRng.startPixelSample(
        CMEM_haltonOwenParams,
        make_int2(kinput.state->pixelCoordX, kinput.state->pixelCoordY),
        kinput.state->sampleIndex);
    // choose a light and prepare BSDF data from intersection
    // TODO maybe: readonly function to fetch 32 bytes with __ldg?
    int const lightIdx =
        min(static_cast<int>(warpRng.get1D(CMEM_haltonOwenParams) * lightCount),
            lightCount - 1);
    Light const light = d_lights[lightIdx];
    BSDF const bsdf = [&] __device__() {  // TODO copy?
      BSDF theBsdf = d_bsdfs[kinput.matId];
      prepareBSDF(&theBsdf, kinput.normal, -kinput.rayD);
      return theBsdf;
    }();

    // cast a shadow ray and perform a anyhit query
    // if hit -> NEE
    float3 Le{0, 0, 0};
    float3 const beta = kinput.state->throughput;
    anyhitNEE(-kinput.rayD, beta, kinput.pos, kinput.error, bsdf, light,
              d_triSoup, kinput.normal, lightCount,
              atomicAdd(&kinput.state->lastBounceTransmission, 0),
              warpRng.get2D(CMEM_haltonOwenParams), &Le);
    // TODO: remove atomics and write normally. State is owned exclusively by k
    atomicAdd(&kinput.state->L.x, Le.x);
    atomicAdd(&kinput.state->L.y, Le.y);
    atomicAdd(&kinput.state->L.z, Le.y);

    AH_PRINT(
        "AH [%u %u] px: [%d %d] d: %d | light: %d BSDF: %d | MIS Weighted "
        "Le: "
        "%f %f %f\n",
        blockIdx.x, threadIdx.x, kinput.state->pixelCoordX,
        kinput.state->pixelCoordY, kinput.state->depth, lightIdx, kinput.matId,
        Le.x, Le.y, Le.z);

    __syncwarp(activeWorkers);  // TODO is this necessary?
    lane = getCoalescedLaneId(activeWorkers);
    mask = inQueue.queuePop(&kinput);
  }
}