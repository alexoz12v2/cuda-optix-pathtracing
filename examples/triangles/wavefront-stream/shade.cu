#include "wave-kernels.cuh"

__global__ void shadeKernel(QueueType<ShadeInput> inQueue,
                            QueueType<ClosestHitInput> outQueue,
                            DeviceArena<PathState> pathStateSlots,
                            float4* d_outBuffer, BSDF* d_bsdfs) {
  ShadeInput input{};
#if USE_SIMPLE_QUEUE
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < inQueue.queueSize(); idx += blockDim.x * gridDim.x) {
    if (!inQueue.marked[idx]) {
      SH_PRINT("SH [%u %u] Queue not marked. Skipping\n", blockIdx.x,
               threadIdx.x);
      continue;
    }
    inQueue.marked[idx] = 0;
    input = inQueue.buffer[idx];

    if ((pathStateSlots.bitmask[input.state->bufferSlot / 32] &
         (1 << (input.state->bufferSlot % 32))) == 0) {
      printf("SH [%u %u] bitmask %x selecting %d\n", blockIdx.x, threadIdx.x,
             pathStateSlots.bitmask[input.state->bufferSlot / 32],
             input.state->bufferSlot % 32);
    }
    assert((pathStateSlots.bitmask[input.state->bufferSlot / 32] &
            (1 << (input.state->bufferSlot % 32))) != 0);
#else
  int mask = inQueue.queuePop<false>(&input);
  int lane = getLaneId();
  while (mask & (1 << lane)) {
#endif
    int const activeWorkers = __activemask();

    int px, py;
    input.state->ldg_px(px, py);

    // Inside each bsdf
    // 1. if max depth reached, kill path
    bool pathDied = false;
    static int constexpr MAX_DEPTH = 32;  // dies afterwards (TODO cmdline)
    // assumes threads have different states
    int const oldDepth = atomicAdd(&input.state->depth, 1);
    // SH_PRINT("SH [%u %u] px: %d %d | d %d | Received object\n", blockIdx.x,
    //          threadIdx.x, px, py, oldDepth);
    if (oldDepth >= MAX_DEPTH) {
      pathDied = true;
    }

    float3 wi{0, 0, 0};
    if (!pathDied) {
      // 2. (not dead) BSDF sampling and bounce computation
      BSDF const bsdf = [&] __device__() {  // TODO copy?
        BSDF theBsdf = d_bsdfs[input.matId];
        prepareBSDF(&theBsdf, input.normal, -input.rayD,
                    load_cv(&input.state->transmissionCount));
        return theBsdf;
      }();
      BSDFSample const bs =
          sampleBsdf(bsdf, -input.rayD, input.normal, input.normal,
                     PcgHash::get2D<float2>(), PcgHash::get1D<float>());
      if (bs) {
        wi = bs.wi;
#if 0
        if (bs.refract)
          printf("SH [%u %u] px: %d %d | d %d | refract %d | woNext %f %f %f\n",
                 blockIdx.x, threadIdx.x, px, py, oldDepth, bs.refract,
                 -bs.wi.x, -bs.wi.y, -bs.wi.z);
#endif
        // TODO: if 1 thread per path, atomics are not necessary
        // 3. (not dead) update path state
        atomicAdd(&input.state->anySpecularBounces, (int)bs.delta);
        atomicExch(&input.state->lastBounceTransmission, (int)bs.refract);
        atomicAdd(&input.state->transmissionCount, (int)bs.refract);
        float3 beta = input.state->throughput * bs.f *
                      fabsf(dot(bs.wi, input.normal)) / bs.pdf;
        // 4. (not dead) russian roulette. If fails, kill path
        if (float const rrBeta =
                fminf(0.95f, maxComponentValue(beta * bs.eta * bs.eta));
            rrBeta < 1 && oldDepth > 1) {
          float const q = fmaxf(0.f, 1.f - rrBeta);
          if (PcgHash::get1D<float>() < q) {
            pathDied = true;
          } else {
            beta /= 1 - q;
            atomicExch(&input.state->throughput.x, beta.x);
            atomicExch(&input.state->throughput.y, beta.y);
            atomicExch(&input.state->throughput.z, beta.z);
          }
        }
      } else {
        pathDied = true;
      }
    }

    if (pathDied) {
      // 5. if path killed, sink to output
      // no need for atomic operation here
      // TODO align
      //// assert(((uintptr_t)&input.state->L) & 15 == 0);
      //// float4 color = *reinterpret_cast<float4*>(&input.state->L);
      //// color.w = 1;  // TODO filter weight?
      float4 const color =
          make_float4(input.state->L.x, input.state->L.y, input.state->L.z, 1);
      // SH_PRINT(
      //     "SH [%u %u]  px [%u %u] d: %d | path died | Bitmask before died
      //     %x\n", blockIdx.x, threadIdx.x, px, py, oldDepth,
      //     pathStateSlots.bitmask[input.state->bufferSlot / 32]);
      assert((pathStateSlots.bitmask[input.state->bufferSlot / 32] &
              (1 << (input.state->bufferSlot % 32))) != 0);
      freeState(pathStateSlots, input.state);
      // SH_PRINT(
      //     "SH [%u %u]  px [%u %u] d: %d | path died | Bitmask after died
      //     %x\n", blockIdx.x, threadIdx.x, px, py, oldDepth,
      //     pathStateSlots.bitmask[input.state->bufferSlot / 32]);
      assert((pathStateSlots.bitmask[input.state->bufferSlot / 32] &
              (1 << (input.state->bufferSlot % 32))) == 0);
      input.state = nullptr;

      // 3. sink to output buffer
      // no need for atomic operation here
      d_outBuffer[py * CMEM_imageResolution.x + px] += color;
      SH_PRINT(
          "SH [%u %u]  px [%u %u] d: %d | path died | L: %f %f %f (w: %f)\n",
          blockIdx.x, threadIdx.x, px, py, oldDepth, color.x, color.y, color.z,
          color.w);
    } else {
      // 6. if path alive, push to closesthit next bounce
      ClosestHitInput const closestHitInput{
          .state = input.state,
          .ray = {
              .o = offsetRayOrigin(input.pos, input.error, input.normal, wi),
              .d = wi,
          }};  // TODO SMEM?
      // SH_PRINT("SH [%u %u] px: %d %d | d %d | Path Survived. Pushing\n",
      //          blockIdx.x, threadIdx.x, px, py, oldDepth);

#if USE_SIMPLE_QUEUE
      bool const pushed = outQueue.queuePush<false>(closestHitInput);
#  ifdef DMT_DEBUG
      assert(pushed);
#  else
      if (getLaneId() == __ffs(__activemask) - 1) {
        printf("SH Failed: closest hit queue full (cap: %d)\n",
               outQueue.capacity);
      }
      asm volatile("trap;");
#  endif
#else
#  ifdef DMT_DEBUG
      int const coalescedLane = getCoalescedLaneId(__activemask());
#  endif
      unsigned const pushMask = outQueue.queuePush<false>(&closestHitInput);
      // SH_PRINT("SH [%u %u]  px [%u %u] d: %d | pushed to closesthit 0x%x\n",
      //          blockIdx.x, threadIdx.x, px, py, oldDepth, pushMask);
#  ifdef DMT_DEBUG
      assert(1u << coalescedLane & pushMask);
#  else
      if (!(1u << coalescedLane & pushMask)) {
        asm volatile("trap;");
      }
#  endif
#endif
    }

    __syncwarp(activeWorkers);  // TODO is this necessary?
#if USE_SIMPLE_QUEUE
#else
    lane = getCoalescedLaneId(activeWorkers);
    mask = inQueue.queuePop<false>(&input);
#endif
  }
}
