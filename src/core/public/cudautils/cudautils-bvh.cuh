#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_BVH_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_BVH_CUH

#include "cudautils/cudautils-macro.cuh"
#include "cudautils/cudautils-vecmath.cuh"

namespace dmt {

struct HitOut {
  float t;
  int triIdx;
};
struct StackEntry {
  int nodeIdx;
  float tmin;
  bool isInner;
};

DMT_GPU bool intersect_tri_mt(Ray const& r, float3 const& v0, float3 const& v1,
                              float3 const& v2, float tMax, float& tOut);

// ---- AABB intersect (per-lane) ----
DMT_FORCEINLINE DMT_GPU bool aabb_intersect_local(Ray const& r, float bxmin,
                                                  float bxmax, float bymin,
                                                  float bymax, float bzmin,
                                                  float bzmax, float tcur,
                                                  float& tminOut) {
  // slabs using precomputed invd
  float tx1 = (bxmin - r.o.x) * r.d_inv.x;
  float tx2 = (bxmax - r.o.x) * r.d_inv.x;
  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);

  float ty1 = (bymin - r.o.y) * r.d_inv.y;
  float ty2 = (bymax - r.o.y) * r.d_inv.y;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float tz1 = (bzmin - r.o.z) * r.d_inv.z;
  float tz2 = (bzmax - r.o.z) * r.d_inv.z;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));

  if (tmax >= tmin && tmin < tcur) {
    tminOut = tmin;
    return true;
  }
  return false;
}
};  // namespace dmt
#endif  // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_BVH_CUH
