#include "light.cuh"

#include "encoding.cuh"
#include "debug.cuh"
#include "sampling.cuh"

#include <cooperative_groups.h>

#include <numbers>

namespace cg = cooperative_groups;

__host__ __device__ LightSample samplePointLight(Light const& light,
                                                 float3 const position,
                                                 float2 u, bool hadTransmission,
                                                 float3 const normal) {
  assert(fabsf(length2(normal) - 1.f) < 1e-3f && "Expected unit vector");
  PRINT("    - Entered Point Light Sampling Procedure\n");
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
  PRINT("    - Light Sample: { p: %f %f %f t: %f pdf: %f }\n", sample.pLight.x,
        sample.pLight.y, sample.pLight.z, sample.distance, sample.pdf);
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
  PRINT("    - Sample Light: Still alive before the coalesced threads\n");
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
      float unused{};
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
// Lights make/eval
// ---------------------------------------------------------------------------
namespace {

__host__ __device__ __forceinline__ void packIntensity(Light& l,
                                                       float3 const color,
                                                       ELightType type) {
  l.intensity[0] = float_to_half_bits(color.x);
  l.intensity[1] = float_to_half_bits(color.y);
  l.intensity[2] = float_to_half_bits(color.z);
  l.intensity[3] = static_cast<uint16_t>(type);
}

}  // namespace

__host__ __device__ Light makePointLight(float3 const color,
                                         float3 const position, float radius) {
  Light l{};
  packIntensity(l, color, ELightType::ePoint);
  l.data.point.pos = position;
  l.data.point.radius = float_to_half_bits(radius);
  return l;
}

__host__ __device__ Light makeSpotLight(float3 color, float3 position,
                                        float3 direction, float cosTheta0,
                                        float cosThetaE, float radius) {
  Light l{};
  packIntensity(l, color, ELightType::eSpot);
  l.data.spot.pos = position;
  l.data.spot.direction = octaFromDir(direction);
  l.data.spot.cosTheta0 = float_to_half_bits(cosTheta0);
  l.data.spot.cosThetaE = float_to_half_bits(cosThetaE);
  l.data.spot.radius = float_to_half_bits(radius);
  return l;
}

__host__ __device__ Light makeDirectionalLight(float3 const color,
                                               float3 const direction,
                                               float const oneMinusCosAngle) {
  Light l{};
  packIntensity(l, color, ELightType::eDirectional);
  l.data.dir.direction = octaFromDir(direction);
  l.data.dir.oneMinusCosAngle = float_to_half_bits(oneMinusCosAngle);
  return l;
}

__host__ __device__ Light makeEnvironmentalLight(float3 const color) {
  Light l{};
  packIntensity(l, color, ELightType::eEnv);
  return l;
}

__host__ __device__ float3 evalLight(Light const& light,
                                     LightSample const& ls) {
  float3 Le = light.getIntensity() * ls.factor;
  if (ELightType const type = light.type();
      type == ELightType::ePoint || type == ELightType::eSpot) {
    // quadratic attenuation (TODO Other variances?)
    // printf("------- Light Before %f %f %f\n", Le.x, Le.y, Le.z);
    Le /= (ls.distance * ls.distance);
    // printf("------- Light After %f %f %f\n", Le.x, Le.y, Le.z);
  }
  return Le;
}

__host__ __device__ float3 evalInfiniteLight(Light const& light, float3 dir,
                                             float* pdf) {
  if (ELightType const type = light.type(); type != ELightType::eEnv) {
    *pdf = 0;
    return make_float3(0, 0, 0);
  }

  float3 const Le = light.getIntensity();
  *pdf = 0.25f * std::numbers::pi_v<float>;
  return Le;
}
