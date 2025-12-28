#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <math_constants.h>
#include <vector_types.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <iostream>
#include <float.h>
#include <numbers>
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
inline __host__ __device__ __forceinline__ float sqrf(float const f) {
  return f * f;
}
// threshold on L∾ norm
inline __host__ __device__ __forceinline__ bool nearZero(const float3 v,
                                                         const float tol) {
  return fabsf(v.x) < tol && fabsf(v.y) < tol && fabsf(v.z) < tol;
}
// small improvement if you know for a fact that your numbers are positive
inline __host__ __device__ __forceinline__ bool nearZeroPos(const float3 v,
                                                            float tol) {
  return v.x < tol && v.y < tol && v.z < tol;
}
inline __host__ __device__ __forceinline__ float luminance(const float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}
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
inline __host__ __device__ __forceinline__ float3& operator/=(float3& v,
                                                              float a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
  return v;
}
inline __host__ __device__ __forceinline__ float3& operator/=(float3& v,
                                                              float3 const a) {
  v.x /= a.x;
  v.y /= a.y;
  v.z /= a.z;
  return v;
}
inline __host__ __device__ __forceinline__ float3 operator*(float3 v, float a) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
inline __host__ __device__ __forceinline__ float3& operator*=(float3& v,
                                                              float a) {
  v.x *= a;
  v.y *= a;
  v.z *= a;
  return v;
}
inline __host__ __device__ __forceinline__ float3 operator*(float a, float3 v) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
inline __host__ __device__ __forceinline__ float2 operator*(float2 v, float a) {
  return make_float2(v.x * a, v.y * a);
}
inline __host__ __device__ __forceinline__ float2 operator*(float a, float2 v) {
  return make_float2(v.x * a, v.y * a);
}
inline __host__ __device__ __forceinline__ float2& operator*=(float2& v,
                                                              float a) {
  v.x = a;
  v.y = a;
  return v;
}
inline __host__ __device__ __forceinline__ float3 operator*(float3 a,
                                                            float3 v) {
  return make_float3(v.x * a.x, v.y * a.y, v.z * a.z);
}
inline __host__ __device__ __forceinline__ float3 operator+(float3 a,
                                                            float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ __forceinline__ float3& operator+=(float3& a,
                                                              float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
inline __host__ __device__ __forceinline__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ __forceinline__ float3 operator+(float b, float3 a) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a,
                                                            float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ __forceinline__ float3 operator-(float a, float3 b) {
  return make_float3(a - b.x, a - b.y, a - b.z);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ __forceinline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ __forceinline__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
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
inline __host__ __device__ __forceinline__ float average(float3 a) {
  return (a.x + a.y + a.z) / 3.f;
}

inline __host__ __device__ __forceinline__ float2 operator+(float2 a,
                                                            float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ __forceinline__ float2 operator-(float2 a,
                                                            float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ __forceinline__ float3 lerp(float3 a, float3 b,
                                                       float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}

inline __host__ __device__ __forceinline__ float sin_sqr_to_one_minus_cos(
    float const s_sq) {
  // Using second-order Taylor expansion at small angles for better accuracy.
  return s_sq > 0.0004f ? 1.0f - safeSqrt(1.0f - s_sq) : 0.5f * s_sq;
}
inline __host__ __device__ __forceinline__ float safeacos(float v) {
  return fminf(fmaxf(acosf(v), -1), 1);
}
inline __host__ __device__ __forceinline__ float length2(float3 v) {
  return dot(v, v);
}
inline __host__ __device__ __forceinline__ float length2(float2 v) {
  return dot(v, v);
}
inline __host__ __device__ __forceinline__ void gramSchmidt(float3 n, float3* a,
                                                            float3* b) {
  assert(a && b && abs(length(n) - 1.f) < 1e-5f);
  if (n.x != n.y || n.x != n.z)
    *a = {n.z - n.y, n.x - n.z, n.y - n.x};  //(1,1,1)x N
  else
    *a = {n.z - n.y, n.x + n.z, -n.y - n.x};  //(-1,1,1)x N

  *a = normalize(*a);
  *b = cross(n, *a);
}
inline __host__ __device__ __forceinline__ void orthonormalTangent(
    const float3 n, float3 const t, float3* a, float3* b) {
  *b = normalize(cross(n, t));
  *a = cross(*b, n);
}
inline __host__ __device__ __forceinline__ float2
cartesianFromPolar(float rho, float phi) {
  return {rho * cosf(phi), rho * sinf(phi)};
}

inline __host__ __device__ __forceinline__ float smoothstep(float x) {
  if (x <= 0.f) return 0.f;
  if (x >= 1.f) return 1.f;
  float const x2 = x * x;
  return 3.f * x2 - 2.f * x2 * x;
}
inline __host__ __device__ __forceinline__ float smoothstep(float a, float b,
                                                            float x) {
  float const t = fmaxf(fminf((x - a) / (b - a), 0.f), 1.f);

  return smoothstep(t);
}
inline __host__ __device__ __forceinline__ float sin_from_cos(
    float const cosine) {
  return safeSqrt(1.f - sqrf(cosine));
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
__host__ __device__ float sphereLightPDF(float distSqr, float radiusSqr,
                                         float3 n, float3 rayD,
                                         bool hadTransmission);
// probably not needed as we are deleting texture coordinates from sample types
__host__ __device__ float2 mapToSphere(float3 co);
__host__ __device__ bool raySphereIntersect(float3 rayO, float3 rayD,
                                            float tMin, float tMax,
                                            float3 sphereC, float sphereRadius,
                                            float3* isect_p, float* isect_t);
__host__ __device__ float3 sampleUniformCone(float3 const N,
                                             float const one_minus_cos_angle,
                                             float2 const rand,
                                             float* cos_theta, float* pdf);
__host__ __device__ float3 sampleUniformSphere(float2 const rand);
__host__ __device__ float2 sampleUniformDisk(float2 u);
__host__ __device__ float3 sampleCosHemisphere(float3 n, float2 u, float* pdf);
__host__ __device__ float3 sampleUniformHemisphere(float3 n, float2 u,
                                                   float* pdf);
__host__ __device__ float cosHemispherePDF(float3 n, float3 d);

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

inline __host__ __device__ __forceinline__ float3
half_vec_to_float3(uint16_t const h[3]) {
  return make_float3(half_bits_to_float(h[0]), half_bits_to_float(h[1]),
                     half_bits_to_float(h[2]));
}

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
    return make_float3(half_bits_to_float(intensity[0]),
                       half_bits_to_float(intensity[1]),
                       half_bits_to_float(intensity[2]));
  }
  __device__ __host__ ELightType type() const {
    return static_cast<ELightType>(intensity[3]);
  }
};
static_assert(sizeof(Light) == 32 && alignof(Light) == 4);

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
inline __host__ __device__ __forceinline__ uint32_t part1by1(uint32_t x) {
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x | (x << 8)) & 0x00ff00ff;  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x | (x << 4)) & 0x0f0f0f0f;  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x | (x << 2)) & 0x33333333;  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x | (x << 1)) & 0x55555555;  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}
inline __host__ __device__ __forceinline__ uint32_t compact1By1(uint32_t x) {
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
  // tri0->matIndex | tri1->matIndex ...
  uint32_t* matId;

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
        matId{},
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
  float3 error;    // intersection error bounds
  int32_t hit;     // 0 = no hit, 1 = hit
  uint32_t matId;  // bsdf index
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

// eta should be flipped accordingly to direction (transmission i->o => 1/eta)
inline __host__ __device__ __forceinline__ float reflectanceFresnelDielectric(
    float cosThetaI, float eta, float* r_cosThetaT) {
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
  *r_cosThetaT = cosThetaT;
  // compute reflectance polarized on the two planes ...
  float const rParl =
      ((eta * cosThetaI) - (cosThetaT)) / ((eta * cosThetaI) + (cosThetaT));
  float const rPerp =
      ((cosThetaI) - (eta * cosThetaT)) / ((cosThetaI) + (eta * cosThetaT));
  // ... then average them
  return (rParl * rParl + rPerp * rPerp) * 0.5f;
}

inline __host__ __device__ __forceinline__ float3
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

// ---------------------------------------------------------------------------
// Software lookup table with linear interpolation
// ---------------------------------------------------------------------------
__host__ __device__ float lookupTableRead(float const* __restrict__ table,
                                          float x, int32_t size);
__host__ __device__ float lookupTableRead2D(float const* __restrict__ table,
                                            float x, float y, int32_t sizex,
                                            int32_t sizey);

// ---------------------------------------------------------------------------
// BSDF
// ---------------------------------------------------------------------------
extern cudaTextureObject_t tex_ggx_E;
extern cudaTextureObject_t tex_ggx_Eavg;

__host__ void allocateDeviceGGXEnergyPreservingTables();
__host__ void freeDeviceGGXEnergyPreservingTables();

enum class EBSDFType : uint16_t {
  eOrenNayar = 0,
  eGGXDielectric = 1,
  eGGXConductor = 2,
};

struct BSDF {
  // its max component is used to weight samples, while the whole RGB value
  // is used in the energy preservation term
  uint16_t weightStorage[4];  // 3xFP16: weight, 1 uint16 -> type
  union BSDFUnion {
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

__host__ __device__ BSDF makeOrenNayar(float3 color, float roughness);
__host__ __device__ BSDF makeGXXDielectric(float3 reflectanceTint,
                                           float3 transmittanceTint, float phi0,
                                           float eta, float alphax,
                                           float alphay);
__host__ __device__ BSDF makeGXXConductor(float3 eta, float3 kappa, float phi0,
                                          float alphax, float alphay);

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

// 1) incident angles from lower hemisphere are invalid. If you are within
// a material after a transmission, it's the caller's responsibility
// to flip the normals so that cosines are positive
// 2) (GGX) eta always refers to outside/inside. hence, if last was
// transmission, the caller should flip it.
__host__ __device__ BSDFSample sampleBsdf(BSDF const& bsdf, float3 wo,
                                          float3 ns, float3 ng, float2 u,
                                          float uc);
__host__ __device__ float3 evalBsdf(BSDF const& bsdf, float3 wo, float3 wi,
                                    float3 ns, float3 ng, float* pdf,
                                    bool* isDelta);

// ---------------------------------------------------------------------------
// BSDF-Specific sampling/evaluation functions
// ---------------------------------------------------------------------------

// Note: Shading normals are not allowed to change the hemisphere of light
// transport. (check and change hemisphere of ns if necessary)

// assumes energy preservation table has been initialized
__host__ __device__ BSDFSample sampleGGX(BSDF const& bsdf, float3 wo, float3 ns,
                                         float3 ng, float2 u, float uc);
__host__ __device__ BSDFSample sampleOrenNayar(BSDF const& bsdf, float3 wo,
                                               float3 ns, float3 ng, float2 u,
                                               float uc);
__host__ __device__ float3 evalGGX(BSDF const& bsdf, float3 wo, float3 wi,
                                   float3 ns, float3 ng, float* pdf,
                                   bool* isDelta);
__host__ __device__ float3 evalOrenNayar(BSDF const& bsdf, float3 wo, float3 wi,
                                         float3 ns, float3 ng, float* pdf,
                                         bool* isDelta);
__host__ __device__ void prepareBSDF(BSDF* bsdf, float3 ns, float3 wo);

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
__global__ void triangleIntersectKernel(TriangleSoup soup, Ray ray,
                                        uint32_t* intersected);
void triangleIntersectTest();

// Large Kernel Parameters from Volta with R530. Otherwise, it's 4096 KB
// https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/

// Grid-Stride Loop + Occupancy API = profit
// https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

__global__ void __launch_bounds__(/*max threads per block*/ 512,
                                  /*min blocks per SM*/ 10)
    basicIntersectionMegakernel(DeviceCamera* d_cam, TriangleSoup d_triSoup,
                                Light const* d_lights,
                                uint32_t const lightCount,
                                Light const* d_infiniteLights,
                                uint32_t const infiniteLightCount,
                                BSDF const* d_bsdf, uint32_t const bsdfCount,
                                DeviceHaltonOwen* d_haltonOwen,
                                float4* d_outBuffer);
