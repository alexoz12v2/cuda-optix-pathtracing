#ifndef DMT_CUDA_CORE_BSDF_CUH
#define DMT_CUDA_CORE_BSDF_CUH

#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/encoding.cuh"

#include <numbers>

enum class EBSDFType : uint16_t {
  eOrenNayar = 0,
  eGGXDielectric = 1,
  eGGXConductor = 2,
  eLambert = 3,
};

struct BSDF {
  // its max component is used to weight samples, while the whole RGB value
  // is used in the energy preservation term
  uint16_t weightStorage[4];  // 3xFP16: weight, 1 uint16 -> type
  union BSDFUnion {
    struct Lambert {
      uint8_t _padding[24];
    } lambert;
    struct OrenNayar {
      uint16_t albedo[3];        // 3xFP16 (TODO check if need)
      uint16_t multiScatter[3];  // 3xFP16
      uint16_t roughness;        // FP16. sigma in radians (TODO check if need)
      uint16_t a;                // FP16. First precomputed term
      uint16_t b;                // FP16. Second precomputed term
      uint8_t _padding[6];
    } orenNayar;
    struct GGX {
      float energyScale;
      uint16_t phi0;    // integer remapped to 0,2π. angle to azimuth from right
      uint16_t alphax;  // integer remapped to 0,1
      uint16_t alphay;  // integer remapped to 0,1
      union UGGXMat {
        struct GGXDielectric {
          uint16_t eta;                   // FP16, IOR
          uint16_t reflectanceTint[3];    // 3xFP16
          uint16_t transmittanceTint[3];  // 3xFP16
        } dielectric;
        struct GGXConductor {
          uint16_t eta[3];    // 3xFP16, real part of complex IOR
          uint16_t kappa[3];  // 3xFP16, complex part of IOR
          uint8_t _padding[2];
        } conductor;
      } mat;

      __host__ __device__ __forceinline__ float getPhi0() const {
        return static_cast<float>(phi0) / UINT16_MAX * 2.f *
               std::numbers::pi_v<float>;
      }
    } ggx;
  } data;

  __host__ __device__ void setWeight(float3 const w) {
    weightStorage[0] = float_to_half_bits(w.x);
    weightStorage[1] = float_to_half_bits(w.y);
    weightStorage[2] = float_to_half_bits(w.z);
#if 0  // first 3 decimal digits from mantissa are preserved
    if (!isZero(w)) {
      float3 decoded = weight();
      printf("  -> setWeight(%f %f %f)->[%hx %hx %hx] %f %f %f\n", w.x, w.y,
             w.z, weightStorage[0], weightStorage[1], weightStorage[2],
             decoded.x, decoded.y, decoded.z);
    }
#endif
  }
  __host__ __device__ float3 weight() const {
    return make_float3(half_bits_to_float(weightStorage[0]),
                       half_bits_to_float(weightStorage[1]),
                       half_bits_to_float(weightStorage[2]));
  }
  __host__ __device__ EBSDFType type() const {
    return static_cast<EBSDFType>(weightStorage[3]);
  }
};
static_assert(sizeof(BSDF) == 32 && alignof(BSDF) == 4);

struct BSDFSample {
  float3 wi;  // sampled incident direction
  float3 f;   // BSDF value for sampled direction and normal
  float pdf;
  float eta;  // 1=reflection. >1=outside:inside. <1=inside:outside
  bool delta;
  bool refract;

  __host__ __device__ operator bool() const {
    return wi.x != 0 && wi.y != 0 && wi.z != 0 && pdf != 0.f;
  }
};

extern cudaTextureObject_t tex_ggx_E;
extern cudaTextureObject_t tex_ggx_Eavg;

__host__ void allocateDeviceGGXEnergyPreservingTables();
__host__ void freeDeviceGGXEnergyPreservingTables();

__host__ __device__ BSDF makeLambert();
__host__ __device__ BSDF makeOrenNayar(float3 color, float roughness);
__host__ __device__ BSDF makeGGXDielectric(float3 reflectanceTint,
                                           float3 transmittanceTint, float phi0,
                                           float eta, float alphax,
                                           float alphay);
__host__ __device__ BSDF makeGGXConductor(float3 eta, float3 kappa, float phi0,
                                          float alphax, float alphay);

// 1) incident angles from lower hemisphere are invalid. If you are within
// a material after a transmission, it's the caller's responsibility
// to flip the normals so that cosines are positive
// 2) (GGX) eta always refers to outside/inside. hence, if last was
// transmission, the caller should flip it.
__host__ __device__ BSDFSample sampleBsdf(BSDF const& bsdf, float3 wo,
                                          float3 ns, float3 ng, float2 u,
                                          float uc);
__host__ __device__ float3 evalBsdf(BSDF const& bsdf, float3 wo, float3 wi,
                                    float3 ns, float3 ng, float* pdf);

// ---------------------------------------------------------------------------
// BSDF-Specific sampling/evaluation functions
// ---------------------------------------------------------------------------

// Note: Shading normals are not allowed to change the hemisphere of light
// transport. (check and change hemisphere of ns if necessary)

__host__ __device__ BSDFSample sampleLambert(BSDF const& bsdf, float3 wo,
                                             float3 ns, float3 ng, float2 u,
                                             float uc);
// assumes energy preservation table has been initialized
__host__ __device__ BSDFSample sampleGGX(BSDF const& bsdf, float3 wo, float3 ns,
                                         float3 ng, float2 u, float uc);
__host__ __device__ BSDFSample sampleOrenNayar(BSDF const& bsdf, float3 wo,
                                               float3 ns, float3 ng, float2 u,
                                               float uc);
__host__ __device__ float3 evalLambert(BSDF const& bsdf, float3 wo, float3 wi,
                                       float3 ns, float3 ng, float* pdf);
__host__ __device__ float3 evalGGX(BSDF const& bsdf, float3 wo, float3 wi,
                                   float3 ns, float3 ng, float* pdf);
__host__ __device__ float3 evalOrenNayar(BSDF const& bsdf, float3 wo, float3 wi,
                                         float3 ns, float3 ng, float* pdf);
__host__ __device__ void prepareBSDF(BSDF* bsdf, float3 ns, float3 wo,
                                     int transmissionCount);

// ---------------------------------------------------------------------------
// BSDF Bits and Pieces
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 reflect(float3 wo, float3 n) {
  return 2.f * dot(wo, n) * n - wo;
}

// eta = eta_i / eta_t
__device__ __forceinline__ bool refract(float3 wi, float3 n, float eta,
                                        float* etap, float3* wt) {
  float cosThetai = dot(wi, n);
  if (cosThetai < 0)  // inside -> outside
  {
    eta = 1.f / eta;
    cosThetai = -cosThetai;
    n = -n;
  }

  // snell: cosThetat = sqrt(1-sin2Thetai / eta2). if radicand is negative,
  // total internal reflection
  float const sin2Thetai = fmaxf(0.f, 1.f - cosThetai * cosThetai);
  float sin2Thetat = sin2Thetai / (eta * eta);
  if (sin2Thetat > 1.f) {
    return false;
  }

  float const cosThetat = safeSqrt(1.f - sin2Thetat);

  *wt = -wi / eta + (cosThetai / eta - cosThetat) * n;

  if (etap) *etap = eta;

  return true;
}

// eta should be flipped accordingly to direction (transmission i->o => 1/eta)
__host__ __device__ __forceinline__ float reflectanceFresnelDielectric(
    float cosThetaI, float eta, float* r_cosThetaT) {
  cosThetaI = fmaxf(-1.f, fminf(1.f, cosThetaI));
  // _warning_ normal and eta should have been flipped by intersection procedure
#if 0
  assert(cosThetaI > 0);
#else
  bool const entering = cosThetaI > 0.f;
  if (!entering) {
    eta = 1.f / eta;
    cosThetaI = fabsf(cosThetaI);
  }
#endif
  float const sinThetaI = safeSqrt(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
  float const sinThetaT = sinThetaI / eta;
  if (sinThetaT >= 1.f) {
    return 1.f;  // total internal reflection
  }
  float const cosThetaT = safeSqrt(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
  *r_cosThetaT = cosThetaT;
  // compute reflectance polarized on the two planes ...
  float const rParl =
      ((eta * cosThetaI) - (cosThetaT)) / ((eta * cosThetaI) + (cosThetaT));
  float const rPerp =
      ((cosThetaI) - (eta * cosThetaT)) / ((cosThetaI) + (eta * cosThetaT));
  // ... then average them
  return (rParl * rParl + rPerp * rPerp) * 0.5f;
}

__host__ __device__ __forceinline__ float3
reflectanceFresnelConductor(float cosThetaI, float3 eta, float3 k) {
  cosThetaI = fmaxf(-1.f, fminf(1.f, cosThetaI));
  float const cosThetaI2 = cosThetaI * cosThetaI;
  float const sinThetaI2 = 1.f - cosThetaI2;
  float3 const eta2 = make_float3(eta.x * eta.x, eta.y * eta.y, eta.z * eta.z);
  float3 const k2 = make_float3(k.x * k.x, k.y * k.y, k.z * k.z);

  float3 const t0 = eta2 - k2 - sinThetaI2;
  float3 const a2plusb2 = sqrt(t0 * t0 + 4.f * eta2 * k2);
  float3 const t1 = a2plusb2 + cosThetaI2;
  float3 const a = sqrt(0.5f * (a2plusb2 + t0));
  float3 const t2 = 2.f * cosThetaI * a;
  float3 const Rs = (t1 - t2) / (t1 + t2);

  float3 const t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
  float3 const t4 = t2 * sinThetaI2;
  float3 const Rp = Rs * (t3 - t4) / (t3 + t4);

  return 0.5f * (Rp + Rs);
}

#endif  // DMT_CUDA_CORE_BSDF_CUH
