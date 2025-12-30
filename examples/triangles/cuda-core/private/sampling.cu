#include "sampling.cuh"

#include "common_math.cuh"

#include <cooperative_groups.h>

#include <numbers>

namespace cg = cooperative_groups;

///  Functions that take a normal n:
///    sampleCosHemisphere
///    sampleUniformHemisphere
///    sampleUniformCone
///  - return directions in world space, oriented around n, using gramSchmidt.
///  Functions that don’t take a normal:
///    sampleUniformSphere
///    sampleUniformDisk
///  - return samples in their canonical local spaces (unit sphere / disk).

__host__ __device__ float sphereLightPDF(float distSqr, float radiusSqr,
                                         float3 n, float3 rayD,
                                         bool hadTransmission) {
  static constexpr float _1Over2Pi = 0.5 / std::numbers::pi_v<float>;
  if (distSqr > radiusSqr)
    return _1Over2Pi / sin_sqr_to_one_minus_cos(radiusSqr / distSqr);
  else
    return hadTransmission ? _1Over2Pi * 0.5f : cosHemispherePDF(n, rayD);
}

__host__ __device__ float2 mapToSphere(float3 co) {
  static constexpr float _1Over2Pi = 0.5 / std::numbers::pi_v<float>;

  float const l = dot(co, co);
  float u;
  float v;
  if (l > 0.0f) {
    if (co.x == 0.0f && co.y == 0.0f) {
      u = 0.0f; /* Otherwise domain error. */
    } else {
      u = (0.5f - atan2f(co.x, co.y) * _1Over2Pi);
    }
    v = 1.0f - safeacos(co.z / sqrtf(l)) * std::numbers::inv_pi_v<float>;
  } else {
    u = v = 0.0f;
  }

  return {u, v};
}

__host__ __device__ bool raySphereIntersect(float3 rayO, float3 rayD,
                                            float tMin, float tMax,
                                            float3 sphereC, float sphereRadius,
                                            float3* isect_p, float* isect_t) {
  // courtesy of cycles
  float3 const d_vec = sphereC - rayO;
  float const r_sq = sphereRadius * sphereRadius;
  float const d_sq = dot(d_vec, d_vec);
  float const d_cos_theta = dot(d_vec, rayD);

  if (d_sq > r_sq && d_cos_theta < 0.0f) {
    // Ray origin outside sphere and points away from sphere.
    return false;
  }

  float const d_sin_theta_sq = length2(d_vec - d_cos_theta * rayD);

  if (d_sin_theta_sq > r_sq) {
    // Closest point on ray outside sphere.
    return false;
  }

  // Law of cosines
  float const t =
      d_cos_theta - copysignf(sqrtf(r_sq - d_sin_theta_sq), d_sq - r_sq);

  if (t > tMin && t < tMax) {
    *isect_t = t;
    *isect_p = rayO + rayD * t;
    return true;
  }

  return false;
}

__host__ __device__ float3 sampleUniformCone(float3 const N,
                                             float const one_minus_cos_angle,
                                             float2 const rand,
                                             float* cos_theta, float* pdf,
                                             int* delta) {
  if (one_minus_cos_angle > 0) {
    float2 xy = sampleUniformDisk(rand);
    float const r2 = length2(xy);

    /* Equivalent to `mix(cos_angle, 1.0f, 1.0f - r2)`. */
    *cos_theta = 1.0f - r2 * one_minus_cos_angle;

    /* Remap disk radius to cone radius, equivalent to `xy *= sin_theta /
     * sqrt(r2)`. */
#if 1
    xy *=
        safeSqrt(r2 * one_minus_cos_angle * (2.0f - r2 * one_minus_cos_angle));
#else
    xy *= safeSqrt(one_minus_cos_angle * (2.0f - one_minus_cos_angle * r2));
#endif

    float const denom = fmaxf(one_minus_cos_angle, 1e-8f);
    *pdf = 0.5f / (std::numbers::pi_v<float> * denom);

    float3 T{};
    float3 B{};
    gramSchmidt(N, &T, &B);
    return xy.x * T + xy.y * B + *cos_theta * N;
  }

  *delta = true;
  *cos_theta = 1.0f;
  *pdf = 1.0f;

  return N;
}

__host__ __device__ float3 sampleUniformSphere(float2 const rand) {
  float const z = 1.0f - 2.0f * rand.x;
  float const r = safeSqrt(1.f - z * z);  // sin from cos
  float const phi = 2 * std::numbers::pi_v<float> * rand.y;

  // polar to cartesian
  float const xCartesian = r * cosf(phi);
  float const yCartesian = r * sinf(phi);

  return {xCartesian, yCartesian, z};
}

__host__ __device__ float2 sampleUniformDisk(float2 u) {
  // remap x,y to -1,1
  float const a = 2.f * u.x - 1.f;
  float const b = 2.f * u.y - 1.f;

  float phi = 0.f;
  float rho = 0.f;
  if (a == 0.f && b == 0.f) return {};

  if (fabsf(a) > fabsf(b)) {
    static constexpr float piOver4 = std::numbers::pi_v<float> / 4;
    rho = a;
    phi = piOver4 * (b / a);
  } else {
    static constexpr float _3piOver4 = 3 * std::numbers::pi_v<float> / 4;
    rho = b;
    phi = _3piOver4 * (a / b);
  }

  return cartesianFromPolar(rho, phi);
}

__host__ __device__ float3 sampleCosHemisphere(float3 n, float2 u, float* pdf) {
  assert(fabsf(length2(n) - 1.f) < 1e-5f && "Expected unit vector");
  float2 const rand = sampleUniformDisk(u);  // sine if n is unit
  // length and frame
  float const cosTheta = safeSqrt(1.f - length2(rand));
  float3 T, B;
  gramSchmidt(n, &T, &B);
  if (pdf) *pdf = cosTheta * std::numbers::inv_pi_v<float>;

  return rand.x * T + rand.y * B + cosTheta * n;
}

__host__ __device__ float3 sampleUniformHemisphere(float3 n, float2 u,
                                                   float* pdf) {
  assert(fabsf(length2(n) - 1.f) < 1e-5f && "Expected unit vector");
  float const z = u.x;
  float const r = safeSqrt(1 - sqrf(z));
  float const phi = 2 * std::numbers::pi_v<float> * u.y;

  if (pdf) *pdf = 0.5f * std::numbers::inv_pi_v<float>;

  float3 T, B;
  gramSchmidt(n, &T, &B);
  return r * cosf(phi) * T + r * sinf(phi) * B + z * n;
}

__host__ __device__ float cosHemispherePDF(float3 n, float3 d) {
  assert(abs(length2(n) - 1.f) < 1e-5f && abs(length2(d) - 1.f) < 1e-5f);
  float const cosTheta = dot(n, d);
  return cosTheta > 0.f ? cosTheta * std::numbers::inv_pi_v<float> : 0.f;
}
