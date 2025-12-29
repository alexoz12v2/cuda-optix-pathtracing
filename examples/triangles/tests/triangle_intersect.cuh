#pragma once

#include "cuda-core/types.cuh"

__global__ void triangleIntersectKernel(TriangleSoup soup, Ray ray,
                                        uint32_t* intersected);
void triangleIntersectTest();
