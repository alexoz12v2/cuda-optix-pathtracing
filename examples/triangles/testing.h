#pragma once

#include <cuda_runtime.h>

#include <vector>

struct Triangle {
  Triangle() = default;
  Triangle(float3 const _v0, float3 const _v1, float3 const _v2)
      : v0(_v0), v1(_v1), v2(_v2) {}
  float3 v0;
  float3 v1;
  float3 v2;
};

std::vector<Triangle> generateSphereMesh(float3 center, float radius,
                                         int latSubdiv, int lonSubdiv);
