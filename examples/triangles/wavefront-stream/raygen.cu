#include "wave-kernels.cuh"

inline __device__ __forceinline__ void raygen(
    RaygenInput const& raygenInput, DeviceArena<PathState>& pathStateSlots,
    DeviceHaltonOwen& warpRng, DeviceHaltonOwenParams const& params,
    ClosestHitInput& out) {
  int2 const pixel = make_int2(raygenInput.px, raygenInput.py);
  CameraSample const cs = getCameraSample(pixel, warpRng, params);
  assert(!out.state);
  int slot = -1;
  slot = pathStateSlots.allocate();
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

__global__ void raygenKernel(DeviceQueue<ClosestHitInput> outQueue,
                             DeviceArena<PathState> pathStateSlots,
                             DeviceHaltonOwen* d_haltonOwen,
                             DeviceCamera* d_cam, int sampleOffset) {
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
  int const numTilesX = ceilDiv(CMEM_imageResolution.x, px);
  int const numTilesY = ceilDiv(CMEM_imageResolution.y, py);
  int const totalTiles = numTilesX * numTilesY;
  int const totalWorkUnits = totalTiles * CMEM_spp;

  // 4. Grid-Stride loop: each warp grabs a work unit (8x4 tile for _1_
  // specific sample)
  for (int i = theGlobalWarpId; i < totalWorkUnits; i += totalWarps) {
    // Decode 1D index into (Tile, Sample)
    assert(i / totalTiles < CMEM_spp);
    int const sampleIdx = i / totalTiles + sampleOffset;
    int const tileIdx = i % totalTiles;

    // Decode tile index into 2D coordinates
    int const tileY = tileIdx / numTilesX;
    int const tileX = tileIdx % numTilesX;

    // 5. Map threads within the warp into pixels in the 8x4 tile (row major)
    int const x = tileX * px + (laneId % px);
    int const y = tileY * py + (laneId / px);

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
          assert(raygenElem.state);
        }
        // check if 32 available places in queue
        unsigned const succ = outQueue.queuePush(&raygenElem);
        RG_PRINT(
            "RG [%u %u] RAYGEN activemask 0x%x generated sample [%d %d] %d\n",
            blockIdx.x, threadIdx.x, succ, x, y, sampleIdx);
        // simplicity: queues are appropriately sized for CMEM_spp
        assert(succ == __activemask());
      }
    }
  }

  RG_PRINT("RG [%u %u] RAYGEN activemask 0x%x finished generation\n",
           blockIdx.x, threadIdx.x, __activemask());
}
