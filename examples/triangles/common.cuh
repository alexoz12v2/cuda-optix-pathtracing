#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <math_constants.h>
#include <vector_types.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <iostream>
#include <float.h>
#include <vector>

inline constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Common Types
// ---------------------------------------------------------------------------
struct Ray {
  float3 o;
  float3 d;
};

struct RayTile {
  __device__ __forceinline__ void addRay(Ray const& ray) {
    int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();
    ox[laneId] = ray.o.x;
    oy[laneId] = ray.o.y;
    oz[laneId] = ray.o.z;
    dx[laneId] = ray.d.x;
    dy[laneId] = ray.d.y;
    dz[laneId] = ray.d.z;
  }

  float ox[WARP_SIZE];
  float oy[WARP_SIZE];
  float oz[WARP_SIZE];
  float dx[WARP_SIZE];
  float dy[WARP_SIZE];
  float dz[WARP_SIZE];
};

struct Transform {
  float m[16];
  float mInv[16];

  __host__ __device__ Transform();
  __host__ __device__ Transform(float const* m);

  // TODO distinguish between point/vector
  __host__ __device__ float3 applyDirection(float3 v) const;

  __host__ __device__ float3 apply(float3 p) const;
  __host__ __device__ float3 applyInverse(float3 p) const;
  __host__ __device__ float3 applyTranspose(float3 n) const;
  __host__ __device__ float3 applyInverseTranspose(float3 n) const;
};

struct CameraSample {
  /// point on the film to which the generated ray should carry radiance
  /// (meaning pixelPosition + random offset)
  float2 pFilm;

  /// scale factor that is applied when the ray�s radiance is added to the image
  /// stored by the film; it accounts for the reconstruction filter used to
  /// filter image samples at each pixel
  float filterWeight;
  unsigned _padding;
};
static_assert(sizeof(CameraSample) == 16);

inline constexpr int NUM_FILM_DIMENSION = 2;

struct DeviceHaltonOwenParams {
  int32_t baseScales[NUM_FILM_DIMENSION];
  int32_t baseExponents[NUM_FILM_DIMENSION];
  int32_t multInvs[NUM_FILM_DIMENSION];
};

// Or switch to Sobol (easier on GPU)
/// warp-wide struct
struct DeviceHaltonOwen {
  static constexpr int MAX_RESOLUTION = 128;
  __host__ __device__ DeviceHaltonOwenParams computeParams(int width,
                                                           int height);

  __device__ void startPixelSample(DeviceHaltonOwenParams const& params, int2 p,
                                   int32_t sampleIndex, int32_t dim = 0);
  __device__ float get1D(DeviceHaltonOwenParams const& params);
  __device__ float2 get2D(DeviceHaltonOwenParams const& params);
  __device__ float2 getPixel2D(DeviceHaltonOwenParams const& params);

  int haltonIndex[WARP_SIZE];  // pixel
  int dimension[WARP_SIZE];    // general value
};

// Camera Space (left-handed):       y+ up, z+ forward, x+ right
// World/Render Space (left-handed): z+ up, y+:forward, x+ right
struct DeviceCamera {
  float focalLength = 20.f;
  float sensorSize = 36.f;
  float3 dir{0.f, 1.f, 0.f};
  int spp = 4;
  float3 pos{0.f, 0.f, 0.f};
  int width = 256;
  int height = 128;
};

// ---------------------------------------------------------------------------
// Common Math
// ---------------------------------------------------------------------------
inline __host__ __device__ __forceinline__ float length(float3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
inline __host__ __device__ __forceinline__ float3 cross(float3 a, float3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// TODO safety assert
inline __host__ __device__ __forceinline__ float3 normalize(float3 a) {
  float const invMag = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  return make_float3(a.x * invMag, a.y * invMag, a.z * invMag);
}
inline __host__ __device__ __forceinline__ float lerp(float a, float b,
                                                      float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}
inline __host__ __device__ __forceinline__ float3 operator/(float3 v, float a) {
  return make_float3(v.x / a, v.y / a, v.z / a);
}
inline __host__ __device__ __forceinline__ float3 operator/(float3 v,
                                                            float3 a) {
  return make_float3(v.x / a.x, v.y / a.y, v.z / a.z);
}
inline __host__ __device__ __forceinline__ float3 operator*(float3 v, float a) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
inline __host__ __device__ __forceinline__ float3 operator*(float a, float3 v) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
inline __host__ __device__ __forceinline__ float3 operator*(float3 a,
                                                            float3 v) {
  return make_float3(v.x * a.x, v.y * a.y, v.z * a.z);
}
inline __host__ __device__ __forceinline__ float3 operator+(float3 a,
                                                            float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ __forceinline__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a,
                                                            float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ __forceinline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ __forceinline__ float safeSqrt(float a) {
  return sqrtf(fmaxf(a, 0.f));
}
inline __host__ __device__ __forceinline__ float3 sqrt(float3 a) {
  return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}
inline __host__ __device__ __forceinline__ float3 abs(float3 a) {
  return make_float3(fabsf(a.x), fabsf(a.y), fabs(a.z));
}

inline __host__ __device__ __forceinline__ float2 operator+(float2 a,
                                                            float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ __forceinline__ float2 operator-(float2 a,
                                                            float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Octahedral mapping (normal buffer, light direction)
// ---------------------------------------------------------------------------
// TODO: probably __forceinline__?
__host__ __device__ uint32_t octaFromDir(float3 const dir);
__host__ __device__ float3 dirFromOcta(uint32_t const octa);

// ---------------------------------------------------------------------------
// Half Precision floats Portable storage
// ---------------------------------------------------------------------------
// on device code, you can use float16 intrinsics. Host coverts back and forth
// from full 4-byte float
// TODO: probably __forceinline__
__host__ __device__ uint16_t float_to_half_bits(float f);
__host__ __device__ float half_bits_to_float(uint16_t h);

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------
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
      // TODO (future) store a pointer/id to texture
      uint8_t _padding[24];
    } env;
    struct Directional {
      uint32_t direction;  // Octahedral mapping
      uint8_t _padding[20];
    } dir;
  } data;

  __device__ __host__ float3 getIntensity() const {
    return make_float3(half_bits_to_float(intensity[0]),
                       half_bits_to_float(intensity[1]),
                       half_bits_to_float(intensity[2]));
  }
  __device__ __host__ ELightType type() const {
    return static_cast<ELightType>(intensity[3]);
  }
};
static_assert(sizeof(Light) == 32 && alignof(Light) == 4);

__host__ __device__ Light makePointLight(float3 const color,
                                         float3 const position, float radius);
// direction assumed normalized
__host__ __device__ Light makeSpotLight(float3 color, float3 position,
                                        float3 direction, float cosTheta0,
                                        float cosThetaE, float radius);
// direction assumed normalized
__host__ __device__ Light makeDirectionalLight(float3 const color,
                                               float3 const direction);
__host__ __device__ Light makeEnvironmentalLight(float3 const color);

struct LightSample {
  float3 pLight;     // sampled point on light source
  float3 direction;  // intersection point to light
  // delta is implicit in light type, which is known
};

// position = last intersection position
__host__ __device__ LightSample sampleLight(Light const& light,
                                            float3 const position,
                                            bool hadTransmission,
                                            float3 const normal, float* pdf);

// ---------------------------------------------------------------------------
// Light specific sampling functions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Morton
// ---------------------------------------------------------------------------
struct MortonLayout2D {
  uint32_t rows;  // height
  uint32_t cols;  // width
  uint32_t mortonRows;
  uint32_t mortonCols;
  uint64_t mortonCount;
};
inline __host__ __device__ __forceinline__ uint32_t nextPow2(uint32_t x) {
  if (x == 0) return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}
inline __host__ __device__ __forceinline__ MortonLayout2D
mortonLayout(uint32_t rows, uint32_t cols) {
  MortonLayout2D layout{};
  layout.rows = rows;
  layout.cols = cols;
  layout.mortonRows = nextPow2(rows);
  layout.mortonCols = nextPow2(cols);
  layout.mortonCount =
      (uint64_t)layout.mortonRows * (uint64_t)layout.mortonCols;
  return layout;
}

// clang-format off
inline
__host__ __device__ __forceinline__ uint32_t part1by1(uint32_t x) {
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x | (x << 8)) & 0x00ff00ff;  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x | (x << 4)) & 0x0f0f0f0f;  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x | (x << 2)) & 0x33333333;  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x | (x << 1)) & 0x55555555;  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}
inline
__host__ __device__ __forceinline__ uint32_t compact1By1(uint32_t x) {
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x | (x >> 1)) & 0x33333333;  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x | (x >> 2)) & 0x0f0f0f0f;  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x | (x >> 4)) & 0x00ff00ff;  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x | (x >> 8)) & 0x0000ffff;  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}
// clang-format on

// row-major morton 2D -> warp 4 x 8
inline __host__ __device__ __forceinline__ uint32_t encodeMorton2D(uint32_t x,
                                                                   uint32_t y) {
  return (part1by1(y) << 1) | part1by1(x);
}
inline __host__ __device__ __forceinline__ void decodeMorton2D(uint32_t code,
                                                               uint32_t* x,
                                                               uint32_t* y) {
  *x = compact1By1(code);
  *y = compact1By1(code >> 1);
}

// ---------------------------------------------------------------------------
// Device Types
// ---------------------------------------------------------------------------
struct TriangleSoup {
  // x0_0 x0_1 x0_2 pad | x1_0 ...
  float* xs;
  float* ys;
  float* zs;
  // array of 4-byte booleans. starts at 0, if intersected 1
  // TODO: change this to u,v,t
  int32_t* intersected;
  // count of triangles
  size_t count;
};

inline __host__ __device__ __forceinline__ float gamma(int32_t n) {
#ifdef __CUDA_ARCH__
  float const f = static_cast<float>(n) * FLT_EPSILON * 0.5f;
#else
  float const f =
      static_cast<float>(n) * std::numeric_limits<float>::epsilon() * 0.5f;
#endif
  return f / (1 - f);
}

inline __host__ __device__ __forceinline__ float3 errorFromTriangleIntersection(
    float u, float v, float3 p0, float3 p1, float3 p2) {
  return gamma(7) * abs(u * p0) + abs(v * p1) + abs((1 - u - v) * p2);
}

struct HitResult {
  __host__ __device__ HitResult()
      : pos{},
        normal{},
        error{},
        hit{},
        t{
#ifdef __CUDA_ARCH__
            CUDART_INF_F
#else
            std::numeric_limits<float>::infinity()
#endif
        } {
  }

  float3 pos;
  float3 normal;
  float3 error;  // intersection error bounds
  int32_t hit;   // 0 = no hit, 1 = hit
  float t;
};

// ---------------------------------------------------------------------------
// Host Types
// ---------------------------------------------------------------------------
struct HostTriangleSoup {
  std::vector<float> xs, ys, zs;
  std::vector<int32_t> expected;
  size_t count = 0;
};

// ---------------------------------------------------------------------------
// CUDA Helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t const err = (call);                                        \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA Error: " << cudaGetErrorName(err) << ": "         \
                << cudaGetErrorString(err) << "\n\tat " << __FILE__ << ':' \
                << __LINE__ << std::endl;                                  \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

struct CudaTimer {
  cudaEvent_t start{}, stop{};

  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~CudaTimer() noexcept {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin(cudaStream_t const stream = 0) const {
    CUDA_CHECK(cudaEventRecord(start, stream));
  }
  float end(cudaStream_t const stream = 0) const {
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ---------------------------------------------------------------------------
// BSDF Bits and Pieces
// ---------------------------------------------------------------------------
inline __device__ __forceinline__ float3 reflect(float3 wo, float3 n) {
  return 2.f * dot(wo, n) * n - wo;
}

// eta = eta_i / eta_t
inline __device__ __forceinline__ bool refract(float3 wi, float3 n, float eta,
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

inline __device__ __forceinline__ float reflectanceFresnelDielectric(
    float cosThetaI, float eta) {
  cosThetaI = fmaxf(-1.f, fminf(1.f, cosThetaI));
  bool const entering = cosThetaI > 0.f;
  if (!entering) {
    eta = 1.f / eta;
    cosThetaI = fabsf(cosThetaI);
  }
  float const sinThetaI = safeSqrt(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
  float const sinThetaT = sinThetaI / eta;
  if (sinThetaT >= 1.f) {
    return 1.f;  // total internal reflection
  }
  float const cosThetaT = safeSqrt(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
  float const rParl =
      ((eta * cosThetaI) - (cosThetaT)) / ((eta * cosThetaI) + (cosThetaT));
  float const rPerp =
      ((cosThetaI) - (eta * cosThetaT)) / ((cosThetaI) + (eta * cosThetaT));
  return (rParl * rParl + rPerp * rPerp) * 0.5f;
}

inline __device__ __forceinline__ float3
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

inline __device__ __forceinline__ float3 sampleGGCX_VNDF(float3 wo, float2 u,
                                                         float ax, float ay) {
  // stretch view (anisotropic roughness)
  float3 const V = normalize(make_float3(ax * wo.x, ay * wo.y, wo.z));

  // orthonormal basis (Frame) (azimuthal only)
  float const lensq = V.x * V.x + V.y * V.y;
  float invLen = rsqrtf(lensq + 1e-7f);

  float3 const T1 = make_float3(-V.y * invLen, V.x * invLen, 0.f);
  float3 const T2 = cross(V, T1);

  // sample disk
  float const r = sqrtf(u.x);
  float const phi = 2.f * CUDART_PI_F * u.y;

  float const t1 = r * cosf(phi);

  // blend toward normal
  float const s = 0.5f * (1.f + V.z);
  float const t2 = lerp(sqrtf(fmaxf(0.f, 1.f - t1 * t1)), r * sinf(phi), s);

  // recombine and unstretch
  float3 Nh = t1 * T1 + t2 * T2 + sqrtf(max(0.f, 1.f - t1 * t1 - t2 * t2)) * V;
  Nh = normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.f, Nh.z)));
  return Nh;
}

__device__ __forceinline__ float smithG1(float3 v, float ax, float ay) {
  float const tan2 = (v.x * v.x) / (ax * ax) + (v.y * v.y) / (ay * ay);
  float const cos2 = v.z * v.z;  // assumes _outward_ normal local space
  return 2.f / (1.f + sqrtf(1.f + tan2 / cos2));
}

// ---------------------------------------------------------------------------
// Host Device Math
// ---------------------------------------------------------------------------
__host__ __device__ Transform worldFromCamera(float3 cameraDirection,
                                              float3 cameraPosition);
__host__ __device__ Transform cameraFromRaster_Perspective(float focalLength,
                                                           float sensorHeight,
                                                           uint32_t xRes,
                                                           uint32_t yRes);

// ---------------------------------------------------------------------------
// Next Event Estimation
// ---------------------------------------------------------------------------
Ray generateShadowRay(float3 const intersection, float3 normal,
                      float3 lightPos);

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ CameraSample getCameraSample(int2 pPixel, DeviceHaltonOwen& rng,
                                        DeviceHaltonOwenParams const& params);
__device__ Ray getCameraRay(CameraSample const& cs,
                            Transform const& cameraFromRaster,
                            Transform const& renderFromCamera);
__global__ void raygenKernel(DeviceCamera* d_cam,
                             DeviceHaltonOwen* d_haltonOwen, RayTile* d_rays);

HitResult hostIntersectMT(const float3& o, const float3& d, const float3& v0,
                          const float3& v1, const float3& v2);
__device__ HitResult triangleIntersect(float4 x, float4 y, float4 z, Ray ray);
__global__ void triangleIntersectKernel(TriangleSoup soup, Ray ray);
void triangleIntersectTest();

__global__ void basicIntersectionMegakernel(DeviceCamera* d_cam,
                                            TriangleSoup d_triSoup,
                                            Light const* d_lights,
                                            uint32_t const lightCount,
                                            DeviceHaltonOwen* d_haltonOwen,
                                            float4* d_outBuffer);