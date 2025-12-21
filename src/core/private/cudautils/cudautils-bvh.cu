#include "cudautils/cudautils-bvh.cuh"

#include "cudautils/cudautils-vecmath.cuh"

namespace dmt {
__forceinline__ __device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 cross3(float3 a, float3 b) {
  float3 x{};
  x.x = a.y * b.z - a.z * b.y;
  x.y = a.z * b.x - a.x * b.z;
  x.z = a.x * b.y - a.y * b.x;
  return x;
}

DMT_GPU bool intersect_tri_mt(Ray const& r, float3 const& v0, float3 const& v1,
                              float3 const& v2, float tMax, float& tOut) {
  float const EPS = 1e-7f;
  float3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
  float3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
  float3 p = cross3(to_float3(r.d), e2);
  float det = dot3(e1, p);
  if (det > -EPS && det < EPS) return false;  // parallel or nearly
  float invDet = 1.0f / det;

  float3 tvec{r.o.x - v0.x, r.o.y - v0.y, r.o.z - v0.z};
  float u = dot3(tvec, p) * invDet;
  if (u < 0.0f || u > 1.0f) return false;

  float3 q = cross3(tvec, e1);
  float v = dot3(to_float3(r.d), q) * invDet;
  if (v < 0.0f || u + v > 1.0f) return false;

  float t = dot3(e2, q) * invDet;
  if (t <= EPS || t >= tMax) return false;

  tOut = t;
  return true;
}
}  // namespace dmt