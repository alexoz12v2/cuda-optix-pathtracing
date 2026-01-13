#include "wave-kernels.cuh"

__device__ __forceinline__ void raygen(RaygenInput const& raygenInput,
                                       DeviceArena<PathState>& pathStateSlots,
                                       DeviceHaltonOwen& warpRng,
                                       DeviceHaltonOwenParams const& params,
                                       ClosestHitInput& out) {
  int2 const pixel = make_int2(raygenInput.px, raygenInput.py);
  CameraSample const cs = getCameraSample(pixel, warpRng, params);
#if DMT_ENABLE_ASSERTS
  assert(!out.state);
#endif
  int slot = -1;
  // RG_PRINT(
  //     "RG [%u %u] RAYGEN activemask 0x%x generated sample [%d %d]. "
  //     "ALlocating\n",
  //     blockIdx.x, threadIdx.x, __activemask(), pixel.x, pixel.y);
  slot = pathStateSlots.allocate();
  // RG_PRINT(
  //     "RG [%u %u] RAYGEN activemask 0x%x generated sample [%d %d]. "
  //     "---------------------------\n",
  //     blockIdx.x, threadIdx.x, __activemask(), pixel.x, pixel.y);
#ifdef DMT_DEBUG
  assert(slot >= 0);
#else
  asm volatile("trap;");
#endif
  out.state = &pathStateSlots.buffer[slot];

  PathState::make(*out.state, raygenInput.px, raygenInput.py,
                  raygenInput.sampleIndex, raygenInput.spp, slot);
  out.ray = getCameraRay(cs, *raygenInput.cameraFromRaster,
                         *raygenInput.renderFromCamera);
}

__global__ void raygenKernel(QueueType<ClosestHitInput> outQueue,
                             DeviceArena<PathState> pathStateSlots,
                             DeviceHaltonOwen* d_haltonOwen,
                             DeviceCamera* d_cam, int tileIdxX, int tileIdxY,
                             int tileDimX, int tileDimY, int sampleOffset) {
  static int constexpr px = 8;
  static int constexpr py = 4;

  DeviceHaltonOwen& warpRng = d_haltonOwen[globalWarpId()];

  int const threadsPerBlock = blockDim.x;
  int const laneId = threadIdx.x % warpSize;
  int const warpsPerBlock = ceilDiv(threadsPerBlock, warpSize);

  // Global warp ID across the raygen portion of the grid
  int const theGlobalWarpId = globalWarpId();
  int const totalWarps = gridDim.x * warpsPerBlock;

  // 3. Define Work Space
  // int const numTilesX = ceilDiv(CMEM_imageResolution.x,
  // CMEM_tileResolution.x); int const numTilesY =
  // ceilDiv(CMEM_imageResolution.y, CMEM_tileResolution.y);
  int const numSubTilesX = ceilDiv(CMEM_tileResolution.x, px);
  int const numSubTilesY = ceilDiv(CMEM_tileResolution.y, py);
  int const totalSubTiles = numSubTilesX * numSubTilesY;
  int const totalWorkUnits = totalSubTiles * CMEM_spp;

  // 4. Grid-Stride loop: each warp grabs a work unit (8x4 tile for _1_
  // specific sample)
  for (int i = theGlobalWarpId; i < totalWorkUnits; i += totalWarps) {
    // Decode 1D index into (Tile, Sample)
#if DMT_ENABLE_ASSERTS
    assert(i / totalSubTiles < CMEM_spp);
#endif
    int const sampleIdx = i / totalSubTiles + sampleOffset;
    int const subTileIdx = i % totalSubTiles;

    // Decode tile index into 2D coordinates
    int const subTileY = subTileIdx / numSubTilesX;
    int const subTileX = subTileIdx % numSubTilesX;

    // 5. Map threads within the warp into pixels in the 8x4 tile (row major)
    int const x = tileDimX * tileIdxX + subTileX * px + (laneId % px);
    int const y = tileDimY * tileIdxY + subTileY * py + (laneId / px);
    RG_PRINT(
        "RG laneid: %d | tileX: %d tileY: %d | subtileX: %d subtileY: %d | "
        "px: %d py: %d\n",
        getLaneId(), tileIdxX, tileIdxY, subTileX, subTileY, x, y);
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
              .spp = d_cam->spp,
              .cameraFromRaster = arrayAsTransform(CMEM_cameraFromRaster),
              .renderFromCamera = arrayAsTransform(CMEM_renderFromCamera),
          };
          warpRng.startPixelSample(CMEM_haltonOwenParams, make_int2(x, y),
                                   sampleIdx);
          raygen(raygenInput, pathStateSlots, warpRng, CMEM_haltonOwenParams,
                 raygenElem);
#if DMT_ENABLE_ASSERTS
          assert(raygenElem.state);
#endif
        }
#if 0
        RG_PRINT(
            "--------------------------------- RG [%u %u] RAYGEN activemask "
            "0x%x generated sample [%d %d] %d | "
            "ks: %d %d | laneid: %d\n",
            blockIdx.x, threadIdx.x, __activemask(), x, y, sampleIdx, i,
            totalWorkUnits, laneId);
#endif

        // check if 32 available places in queue
#if USE_SIMPLE_QUEUE
        bool const pushed = outQueue.queuePush<false>(raygenElem);
#  ifdef DMT_DEBUG
        assert(pushed);
#  else
        if (!pushed) {
          asm volatile("trap;");
        }
#  endif
#else
#  ifdef DMT_DEBUG
        int const coalescedLane = getCoalescedLaneId(__activemask());
#  endif
        unsigned const succ = outQueue.queuePush<false>(&raygenElem);
        // RG_PRINT(
        //     "RG [%u %u] RAYGEN activemask 0x%x generated sample [%d %d] %d |
        //     " "ks: %d %d | laneid: %d\n", blockIdx.x, threadIdx.x, succ, x,
        //     y, sampleIdx, i, totalWorkUnits, laneId);
        // simplicity: queues are appropriately sized for CMEM_spp
#  ifdef DMT_DEBUG
        if (!((1u << coalescedLane) & succ)) {
          int volatile head = *outQueue.head & (outQueue.capacity - 1);
          int volatile tail = *outQueue.tail & (outQueue.capacity - 1);
          RG_PRINT(
              "\033[31mRG [%u %u] RAYGEN activemask 0x%x | push failed: h: %d "
              "t: %d\033[0m\n",
              blockIdx.x, threadIdx.x, __activemask(), head, tail);
        } else {
          int volatile head = *outQueue.head & (outQueue.capacity - 1);
          int volatile tail = *outQueue.tail & (outQueue.capacity - 1);
          RG_PRINT(
              "RG [%u %u] RAYGEN activemask 0x%x | pushed: h: %d t: %d | "
              "remaining: %d\n",
              blockIdx.x, threadIdx.x, __activemask(), head, tail,
              outQueue.capacity - (tail - head));
        }
        assert((1u << coalescedLane) & succ);
#  endif
#endif
      }
    }
  }

  RG_PRINT("RG [%u %u] RAYGEN activemask 0x%x finished generation\n",
           blockIdx.x, threadIdx.x, __activemask());
}
