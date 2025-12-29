#pragma once

#include "cuda-core/types.cuh"
#include "cuda-core/rng.cuh"

struct TestHoltonOwenVals {
  float2 valGet2D;
  float2 valGetPixel2D;
  float valGet1D;
};

__global__ void testForHaltonOwenKernel(DeviceHaltonOwen* d_haltonOwen);
void testForHaltonOwenRunner();
