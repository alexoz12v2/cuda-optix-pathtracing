#include "wave-kernels.cuh"

__global__ void missKernel(DeviceQueue<MissInput> inQueue,
                           DeviceArena<PathState> pathStateSlots,
                           float4* d_outBuffer, DeviceHaltonOwen* d_haltonOwen,
                           Light* infiniteLights, uint32_t infiniteLightCount) {
  // Sink: TODO SMEM array to cache PathStates which are still used?
  DeviceHaltonOwen& warpRng = d_haltonOwen[globalWarpId()];
  MissInput kinput{};

  int lane = getLaneId();
  int mask = inQueue.queuePop(&kinput);
  while (mask & (1 << lane)) {
    int const activeWorkers = __activemask();

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
                             infiniteLightCount),
            infiniteLightCount - 1);
    Light const light = infiniteLights[lightIndex];
    // 2. add to state radiance
    float pdf = 0;
    float3 Le = evalInfiniteLight(light, kinput.rayDirection, &pdf);
    if (pdf) {
      float3 const beta = kinput.state->throughput;
      Le = beta * Le * infiniteLightCount;
      // TODO: If any specular bounce or depth == 0, don't do MIS
      // TODO: Use last BSDF for MIS
      atomicAdd(&kinput.state->L.x, Le.x);
      atomicAdd(&kinput.state->L.y, Le.y);
      atomicAdd(&kinput.state->L.z, Le.z);
    }
    // 2. free state TODO align that thing to 16
    // float4 color = *reinterpret_cast<float4*>(&kinput.state->L);
    float4 const color =
        make_float4(kinput.state->L.x, kinput.state->L.y, kinput.state->L.z, 1);
    // color.w = 1;  // TODO filter weight?
    freeState(pathStateSlots, kinput.state);
    kinput.state = nullptr;

    // 3. sink to output buffer
    // no need for atomic operation here
    d_outBuffer[py * CMEM_imageResolution.x + px] += color;
    MS_PRINT("MS [%u %u] px: [%d %d] | c: %f %f %f\n", blockIdx.x, threadIdx.x,
             px, py, color.x, color.y, color.z);

    __syncwarp(activeWorkers);  // TODO is this necessary?
    lane = getCoalescedLaneId(activeWorkers);
    mask = inQueue.queuePop(&kinput);
  }
}
