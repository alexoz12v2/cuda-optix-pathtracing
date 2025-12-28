#include "common.cuh"

#include <cooperative_groups.h>

#include <numbers>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Sampling Utils
// ---------------------------------------------------------------------------
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
    v = 1.0f - safeacos(co.z / sqrtf(l)) * _1Over2Pi;
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
                                             float* cos_theta, float* pdf) {
  if (one_minus_cos_angle > 0) {
    float2 xy = sampleUniformDisk(rand);
    float const r2 = length2(xy);

    /* Equivalent to `mix(cos_angle, 1.0f, 1.0f - r2)`. */
    *cos_theta = 1.0f - r2 * one_minus_cos_angle;

    /* Remap disk radius to cone radius, equivalent to `xy *= sin_theta /
     * sqrt(r2)`. */
    xy *= safeSqrt(one_minus_cos_angle * (2.0f - one_minus_cos_angle * r2));

    *pdf = 0.5f / (std::numbers::pi_v<float> * one_minus_cos_angle);

    float3 T{};
    float3 B{};
    gramSchmidt(N, &T, &B);
    return xy.x * T + xy.y * B + *cos_theta * N;
  }

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

  if (a > b) {
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
  assert(abs(length2(n) - 1.f) < 1e-5f && "Expected unit vector");
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
  assert(abs(length2(n) - 1.f) < 1e-5f && "Expected unit vector");
  float2 xy = sampleUniformDisk(u);
  float const z = 1.f - length2(xy);

  xy *= safeSqrt(z + 1.f);
  float3 T{}, B{};
  gramSchmidt(n, &T, &B);

  float3 wo = xy.x * T + xy.y * B + z * n;
  if (pdf) *pdf = 0.5f * std::numbers::inv_pi_v<float>;
  return wo;
}

__host__ __device__ float cosHemispherePDF(float3 n, float3 d) {
  assert(abs(length2(n) - 1.f) < 1e-5f && abs(length2(d) - 1.f) < 1e-5f);
  float const cosTheta = dot(n, d);
  return cosTheta > 0.f ? cosTheta * std::numbers::inv_pi_v<float> : 0.f;
}

// ---------------------------------------------------------------------------
// Light specific sampling functions
// ---------------------------------------------------------------------------
__host__ __device__ LightSample samplePointLight(Light const& light,
                                                 float3 const position,
                                                 float2 u, bool hadTransmission,
                                                 float3 const normal) {
  LightSample sample = {};
  sample.factor = 1.0f;  // light intensity doesn't depend on direction

  float const radiusSqr = half_bits_to_float(light.data.point.radius) *
                          half_bits_to_float(light.data.point.radius);
  float3 lightN = position - light.data.point.pos;
  float const distSqr = dot(lightN, lightN);
  float const dist = sqrtf(distSqr);
  lightN /= dist;
  bool const effectivelyDelta =
      (half_bits_to_float(light.data.point.radius) / dist) < 1e-3f;
  float cosTheta = 0.f;

  if (distSqr > radiusSqr) {
    // outside sphere
    float const oneMinusCos = sin_sqr_to_one_minus_cos(radiusSqr / distSqr);
    sample.direction =
        sampleUniformCone(-lightN, oneMinusCos, u, &cosTheta, &sample.pdf);
    if (effectivelyDelta) {
      sample.pdf = 1;
      sample.delta = true;
    }
  } else {
    // inside sphere
    if (hadTransmission) {
      sample.direction = sampleUniformSphere(u);
      sample.pdf = 0.25f * std::numbers::inv_pi_v<float>;
    } else {
      sample.direction = sampleCosHemisphere(normal, u, &sample.pdf);
    }
    cosTheta = -dot(sample.direction, lightN);
  }
  // law of cosines ( a = b cosGamma +- sqrt(c2 - b2 sin2Gamma) )
  // (you know center - pSurf distance and angle between center - pSurf and
  // pSurf - pLight, and want to know distance)
  sample.distance =
      dist * cosTheta -
      copysignf(safeSqrt(radiusSqr - distSqr + distSqr * cosTheta * cosTheta),
                distSqr - radiusSqr);
  // remap sampled point to sphere to prevent precision issues on small radius
  sample.pLight = position + sample.direction * sample.distance;
  float3 const ng = normalize(sample.pLight - light.data.point.pos);
  sample.pLight =
      ng * half_bits_to_float(light.data.point.radius) * light.data.point.pos;
#if !NO_LIGHT_SAMPLE_UV
  // texture coordinates
  Point2f const uv = mapToSphere(light.lightFromRender(normalFrom(sample->ng)));

  // remap to barycentric coords
  sample->uv[0] = uv.y;
  sample->uv[1] = 1.f - uv.x - uv.y;
#endif
  return sample;
}

__host__ __device__ Ray spotLightToLocal(float3 lightPos, float3 lightDirection,
                                         Ray globalSpaceRay) {
  // Build spotlight basis
  float3 forward = normalize(lightDirection);

  // Choose a stable up vector
  float3 up = (fabsf(forward.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f)
                                          : make_float3(0.f, 1.f, 0.f);

  float3 right = normalize(cross(up, forward));
  up = cross(forward, right);

  // Translate ray origin into light space
  float3 o = make_float3(globalSpaceRay.o.x - lightPos.x,
                         globalSpaceRay.o.y - lightPos.y,
                         globalSpaceRay.o.z - lightPos.z);

  // Apply inverse rotation (transpose of basis)
  Ray localRay;
  localRay.o = make_float3(dot(o, right), dot(o, up), dot(o, forward));

  localRay.d =
      make_float3(dot(globalSpaceRay.d, right), dot(globalSpaceRay.d, up),
                  dot(globalSpaceRay.d, forward));

  return localRay;
}

__host__ __device__ LightSample sampleSpotLight(Light const& light,
                                                float3 const position, float2 u,
                                                bool hadTransmission,
                                                float3 const normal) {
  LightSample sample{};
  sample.distance = FLT_MAX;
  float const radiusSqr = half_bits_to_float(light.data.spot.radius) *
                          half_bits_to_float(light.data.spot.radius);
  float3 lightN = position - light.data.spot.pos;
  float const distSqr = dot(lightN, lightN);
  float const dist = sqrtf(distSqr);
  lightN /= dist;
  bool const effectivelyDelta =
      (half_bits_to_float(light.data.spot.radius) / dist) < 1e-3f;
  bool outsideConeRange = false;
  float cosTheta = 0.f;
  if (distSqr > radiusSqr) {
    // if outside sphere
    float const oneMinusCosHalfSpotSpread =
        1.f - half_bits_to_float(light.data.spot.cosThetaE);
    float const oneMinusCosHalfAngle =
        sin_sqr_to_one_minus_cos(radiusSqr / distSqr);
    if (oneMinusCosHalfAngle < oneMinusCosHalfSpotSpread) {
      // direction towards apex: sample visible part of the sphere
      sample.direction = sampleUniformCone(-lightN, oneMinusCosHalfAngle, u,
                                           &cosTheta, &sample.pdf);
    } else {
      // direction towards cone: sample spread cone, if you fall within it
      sample.direction = sampleUniformCone(
          -dirFromOcta(light.data.spot.direction), oneMinusCosHalfSpotSpread, u,
          &cosTheta, &sample.pdf);
      if (!raySphereIntersect(position, sample.direction, 0.f, FLT_MAX,
                              light.data.spot.pos,
                              half_bits_to_float(light.data.spot.radius),
                              &sample.pLight, &sample.distance)) {
        outsideConeRange = true;
        sample.pdf = 0;
      }
    }
  } else {
    // inside sphere
    if (hadTransmission) {
      sample.direction = sampleUniformSphere(u);
      sample.pdf = 0.25f * std::numbers::inv_pi_v<float>;
    } else {
      sample.direction = sampleCosHemisphere(normal, u, &sample.pdf);
    }
    cosTheta = -dot(sample.direction, lightN);
  }

  Ray const localRay = spotLightToLocal(light.data.spot.pos,
                                        dirFromOcta(light.data.spot.direction),
                                        {position, -sample.direction});
  if (!outsideConeRange) {
    // angular attenuation
    sample.factor = spotLightAttenuation(
        localRay.d.z, half_bits_to_float(light.data.spot.cosTheta0),
        half_bits_to_float(light.data.spot.cosThetaE));
    if (sample.factor <= 0.f) {
      outsideConeRange = true;
    }
  }

  if (!outsideConeRange) {
    // if raySphereIntersect branch wasn't executed, we didn't compute distance
    if (sample.distance == FLT_MAX) {
      // law of cosines (see point light for more detailed comment)
      sample.distance = dist * cosTheta *
                        copysignf(safeSqrt(radiusSqr - distSqr +
                                           distSqr * cosTheta * cosTheta),
                                  distSqr - radiusSqr);
      sample.pLight = position + sample.direction * sample.distance;
    }

    if (effectivelyDelta) {
      sample.pdf = 1.f;
      sample.delta = true;
    }

    // remap onto sphere to prevent precision issue with small radius
    sample.pLight = position + sample.direction * sample.distance;
    float3 const ng = normalize(sample.pLight - light.data.spot.pos);
    sample.pLight =
        ng * half_bits_to_float(light.data.spot.radius) * light.data.spot.pos;

#if !NO_LIGHT_SAMPLE_UV
    sample->uv = spotLightUV(
        localRayDir, halfCotHalfSpotAngle(light.data.spot.cosHalfSpotAngle));
#endif
  }
  return sample;
}

// ---------------------------------------------------------------------------
// Light Sampling dispatcher
// ---------------------------------------------------------------------------
// TODO is it ok to synchronize the coalesced warp?
__host__ __device__ LightSample sampleLight(Light const& light,
                                            float3 const position, float2 u,
                                            bool hadTransmission,
                                            float3 const normal) {
  LightSample sample{};
  // on device,we want to enter and exit with the same warp configuration
#ifdef __CUDA_ARCH__
  cg::coalesced_group theWarp = cooperative_groups::coalesced_threads();
#endif
  switch (light.type()) {
    case ELightType::ePoint:
      sample = samplePointLight(light, position, u, hadTransmission, normal);
      break;
    case ELightType::eSpot:
      sample = sampleSpotLight(light, position, u, hadTransmission, normal);
      break;
    case ELightType::eEnv:
      sample.direction = sampleUniformSphere(u);
      sample.pdf = 0.25f * std::numbers::inv_pi_v<float>;
      sample.factor = 1.f;
      sample.pLight = sample.direction;  // Special Case
      sample.distance = FLT_MAX;
      break;
    case ELightType::eDirectional: {
      float unused;
      sample.pLight =  // special case
          sampleUniformCone(dirFromOcta(light.data.dir.direction),
                            half_bits_to_float(light.data.dir.oneMinusCosAngle),
                            u, &unused, &sample.pdf);
      sample.direction = -sample.pLight;
      sample.factor = 1.f;
      sample.delta = true;
      sample.distance = FLT_MAX;
      break;
    }
  }

#ifdef __CUDA_ARCH__
  theWarp.sync();
#endif
  return sample;
}

// ---------------------------------------------------------------------------
// Software lookup table with linear interpolation
// ---------------------------------------------------------------------------
__host__ __device__ float lookupTableRead(float const* __restrict__ table,
                                          float x, int32_t size) {
  x = fminf(fmaxf(x, 0.f), 1.f) * (size - 1);

  int32_t const index = fminf(static_cast<int32_t>(x), size - 1);
  int32_t const nIndex = fminf(index + 1, size - 1);
  float const t = x - index;

  // lerp formula
  float const data0 = table[index];
  if (t == 0.f) return data0;

  float const data1 = table[nIndex];
  return (1.f - t) * data0 + t * data1;
}

__host__ __device__ float lookupTableRead2D(float const* __restrict__ table,
                                            float x, float y, int32_t sizex,
                                            int32_t sizey) {
  y = fminf(fmaxf(y, 0.f), 1.f) * (sizey - 1);

  int32_t const index = fminf(static_cast<int32_t>(y), sizey - 1);
  int32_t const nIndex = fminf(index + 1, sizey - 1);
  float const t = y - index;

  // bilinear interp formula
  float const data0 = lookupTableRead(table + sizex * index, x, sizex);
  if (t == 0.f) return data0;

  float const data1 = lookupTableRead(table + sizex * nIndex, x, sizex);
  return (1.f - t) * data0 + t * data1;
}
