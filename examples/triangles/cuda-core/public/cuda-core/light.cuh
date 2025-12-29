#ifndef DMT_CUDA_CORE_LIGHT_CUH
#define DMT_CUDA_CORE_LIGHT_CUH

#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/encoding.cuh"

enum class ELightType : uint16_t { ePoint, eSpot, eEnv, eDirectional };

struct Light {
  uint16_t intensity[4];  // 3x FP16 + light type as last (high on little end.)

  union UnionLight {
    struct Point {
      float3 pos;
      uint16_t radius;  // FP16, nucleus radius (sampling)
      uint8_t _padding[8];
    } point;
    struct Spot {
      float3 pos;
      uint32_t direction;  // Octahedral mapping
      uint16_t cosTheta0;  // FP16, cosine of maximum intensity angle
      uint16_t cosThetaE;  // FP16, cosine of maximum penumbra angle
      uint16_t radius;     // FP16, nucleus radius (sampling)
      uint8_t _padding[2];
    } spot;
    struct Environmental {
      // TODO (future) store a pointer/id to texture?
      uint8_t _padding[24];
    } env;
    struct Directional {
      uint32_t direction;         // Octahedral mapping
      uint16_t oneMinusCosAngle;  // falloff angle for light spread
      uint8_t _padding[18];
    } dir;
  } data;

  __device__ __host__ float3 getIntensity() const {
    float3 const f3 = make_float3(half_bits_to_float(intensity[0]),
                                  half_bits_to_float(intensity[1]),
                                  half_bits_to_float(intensity[2]));
    // assert(!nearZero(f3, 1e-4f));
    return f3;
  }
  __device__ __host__ ELightType type() const {
    return static_cast<ELightType>(intensity[3]);
  }
};
static_assert(sizeof(Light) == 32 && alignof(Light) == 4);

struct LightSample {
  float3 pLight;     // sampled point on light source
  float3 direction;  // intersection point to light
  // TODO never used in our code?
  // float3 normal; // normal direction of irradiance of light
  float pdf;
  int32_t delta;  // 0 -> not delta, 1 -> delta
#define NO_LIGHT_SAMPLE_UV 1
  // TODO uv?
  // float2 uv;
  float distance;  // t, used in attenuation
  float factor;    // used in spotlight angular attenuation

  // either this or check that PDF is not 0
  __host__ __device__ __forceinline__ operator bool() const {
    return direction.x != 0 && direction.y != 0 && direction.z != 0 && pdf != 0;
  }
};

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------
// compute inverse transform from position and direction and return local ray
__host__ __device__ Ray spotLightToLocal(float3 lightPos, float3 lightDirection,
                                         Ray globalSpaceRay);
// compute angular attenuation
inline __host__ __device__ __forceinline__ float spotLightAttenuation(
    float const cosTheta, float const cosTheta0, float const cosThetaE) {
  // you can multiply cosTheta to customize falloff
  return smoothstep(cosThetaE, cosTheta0, cosTheta);
}

__host__ __device__ Light makePointLight(float3 const color,
                                         float3 const position, float radius);
// direction assumed normalized
__host__ __device__ Light makeSpotLight(float3 color, float3 position,
                                        float3 direction, float cosTheta0,
                                        float cosThetaE, float radius);
// direction assumed normalized
__host__ __device__ Light makeDirectionalLight(float3 const color,
                                               float3 const direction,
                                               float const oneMinusCosAngle);
__host__ __device__ Light makeEnvironmentalLight(float3 const color);

// position = last intersection position
__host__ __device__ LightSample sampleLight(Light const& light,
                                            float3 const position, float2 u,
                                            bool hadTransmission,
                                            float3 const normal);
// ---------------------------------------------------------------------------
// Light specific sampling functions
// ---------------------------------------------------------------------------
__host__ __device__ LightSample samplePointLight(Light const& light,
                                                 float3 const position,
                                                 float2 u, bool hadTransmission,
                                                 float3 const normal);
__host__ __device__ LightSample sampleSpotLight(Light const& light,
                                                float3 const position, float2 u,
                                                bool hadTransmission,
                                                float3 const normal);

// ---------------------------------------------------------------------------
// Light Evaluation
// ---------------------------------------------------------------------------
__host__ __device__ float3 evalLight(Light const& light, LightSample const& ls);

__host__ __device__ float3 evalInfiniteLight(Light const& light, float3 dir,
                                             float* pdf);

#endif  // DMT_CUDA_CORE_LIGHT_CUH