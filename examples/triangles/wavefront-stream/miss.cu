#include "wave-kernels.cuh"

__global__ void missKernel(QueueType<MissInput> inQueue,
                           DeviceArena<PathState> pathStateSlots,
#if DMT_ENABLE_MSE
                           DeviceOutputBuffer d_out,
#else
                           float4* d_outBuffer,
#endif
                           Light* infiniteLights, uint32_t infiniteLightCount) {
  // Sink: TODO SMEM array to cache PathStates which are still used?
  MissInput kinput{};

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

    // TODO if malloc aligned, use int4
    int const px = __ldg(&kinput.state->pixelCoordX);
    int const py = __ldg(&kinput.state->pixelCoordY);

    // 1. choose an infinite light, fetch last BSDF intersection (TODO)
    int const lightIndex =
        min(static_cast<int>(PcgHash::get1D<float>() * infiniteLightCount),
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
#if FORCE_ATOMIC_OPS
      atomicAdd(&kinput.state->L.x, Le.x);
      atomicAdd(&kinput.state->L.y, Le.y);
      atomicAdd(&kinput.state->L.z, Le.z);
#else
      kinput.state->L = Le;
#endif
    }
    // 2. free state TODO align that thing to 16
    // float4 color = *reinterpret_cast<float4*>(&kinput.state->L);
#if DMT_ENABLE_MSE
    float3 const color(kinput.state->L.x, kinput.state->L.y, kinput.state->L.z);
#else
    float4 const color =
        make_float4(kinput.state->L.x, kinput.state->L.y, kinput.state->L.z, 1);
#endif
    // 3. sink to output buffer
    // no need for atomic operation here
#if DMT_ENABLE_MSE
#  if 1
    float const num = kinput.state->sampleIndex + 1;
    float4 L = d_out.meanPtr[py * CMEM_imageResolution.x + px];
    float4 d2 = d_out.m2Ptr[py * CMEM_imageResolution.x + px];

    // welford update
    float3 mean(L.x, L.y, L.z);
    float3 m2(d2.x, d2.y, d2.z);

    float3 delta = color - mean;
    mean += delta / num;
    float3 delta2 = color - mean;
    m2 += delta * delta2;

    L.x = mean.x;
    L.y = mean.y;
    L.z = mean.z;
    d2.x = m2.x;
    d2.y = m2.y;
    d2.z = m2.z;
    d2.w = num;

    // write back (TODO: Switch to sum and sum2, host does variance computation
    // (miss and shade might race)?)
    d_out.meanPtr[py * CMEM_imageResolution.x + px] = L;
    d_out.m2Ptr[py * CMEM_imageResolution.x + px] = d2;
#  else
    float3 const color4(color.x, color.y, color.z, 1);
    atomicAdd(d_out.meanPtr + py * CMEM_imageResolution.x + px, color4);
    atomicAdd(d_out.m2Ptr + py * CMEM_imageResolution.x + px, color4 * color4);
#  endif
#else
    d_outBuffer[py * CMEM_imageResolution.x + px] += color;
#endif
    MS_PRINT("MS [%u %u] px: [%d %d] | c: %f %f %f (w: %f)\n", blockIdx.x,
             threadIdx.x, px, py, color.x, color.y, color.z, color.w);

    freeState(pathStateSlots, kinput.state);
    kinput.state = nullptr;

    __syncwarp(activeWorkers);  // TODO is this necessary?
#if USE_SIMPLE_QUEUE
#else
    lane = getCoalescedLaneId(activeWorkers);
    mask = inQueue.queuePop<false>(&kinput);
#endif
  }
}
