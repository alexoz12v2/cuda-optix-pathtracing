#include "extra_math.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ CameraSample getCameraSample(int2 pPixel, DeviceHaltonOwen& rng,
                                        DeviceHaltonOwenParams const& params) {
  CameraSample cs{};

  // TODO filter result from getPixel2D
  float2 const pixelShift = rng.getPixel2D(params) - make_float2(0.5f, 0.5f);
#if 0
  printf("  pixelShift: %f %f\n", pixelShift.x, pixelShift.y);
#endif

  cs.pFilm =
      pixelShift + make_float2(0.5, 0.5) + make_float2(pPixel.x, pPixel.y);
#if 0
  printf("  pFilm: %f %f\n", cs.pFilm.x, cs.pFilm.y);
#endif
  cs.filterWeight = 1.f;  // TODO filtering
  return cs;
}

__device__ Ray getCameraRayNonRandom(int2 pixel,
                                     Transform const& cameraFromRaster,
                                     Transform const& renderFromCamera) {
  Ray ray;
  float2 const thePixel = make_float2(pixel.x + 0.5f, pixel.y + 0.5f);
  float3 const pCamera =
      cameraFromRaster.apply(make_float3(thePixel.x, thePixel.y, 0.0f));

  ray.o = renderFromCamera.apply(make_float3(0.0f, 0.0f, 0.0f));
  ray.d = normalize(renderFromCamera.applyDirection(pCamera));

  return ray;
}

__device__ Ray getCameraRay(CameraSample const& cs,
                            Transform const& cameraFromRaster,
                            Transform const& renderFromCamera) {
  Ray ray;
  float3 const pCamera =
      cameraFromRaster.apply(make_float3(cs.pFilm.x, cs.pFilm.y, 0.0f));

  ray.o = renderFromCamera.apply(make_float3(0.0f, 0.0f, 0.0f));
  ray.d = normalize(renderFromCamera.applyDirection(pCamera));

  return ray;
}
__host__ __device__ Transform worldFromCamera(float3 cameraDirection,
                                              float3 cameraPosition) {
  float3 forward = normalize(cameraDirection);
  float3 right = normalize(cross(forward, make_float3(0, 0, 1)));
  float3 up = cross(right, forward);

  // Column-major matrix for worldFromCamera
  // clang-format off
  float m[16];
  m[ 0] = right.x; m[ 4] = up.x; m[ 8] = forward.x; m[12] = cameraPosition.x;
  m[ 1] = right.y; m[ 5] = up.y; m[ 9] = forward.y; m[13] = cameraPosition.y;
  m[ 2] = right.z; m[ 6] = up.z; m[10] = forward.z; m[14] = cameraPosition.z;
  m[ 3] = 0.0f;    m[ 7] = 0.0f; m[11] = 0.0f;      m[15] = 1.0f;
  // clang-format on
  return Transform{m};
}

__host__ __device__ Transform cameraFromRaster_Perspective(
    float focalLength_mm, float sensorHeight_mm, uint32_t xRes, uint32_t yRes) {
  // Compute sensor width from aspect ratio
  const float sensorWidth_mm =
      sensorHeight_mm * static_cast<float>(xRes) / static_cast<float>(yRes);

  static constexpr float MM_TO_M = 0.001f;

  float const focalLength_m = focalLength_mm * MM_TO_M;
  float const sensorHeight_m = sensorHeight_mm * MM_TO_M;
  float const sensorWidth_m = sensorWidth_mm * MM_TO_M;

  // Pixel size in mm on the sensor
  const float pixelSizeX = sensorWidth_m / static_cast<float>(xRes);
  const float pixelSizeY = sensorHeight_m / static_cast<float>(yRes);

  // Offset from raster origin (top-left) to sensor coordinates (centered)
  float const tx = -0.5f * sensorWidth_m + 0.5f * pixelSizeX;
  float const ty = 0.5f * sensorHeight_m - 0.5f * pixelSizeY;
  float const tz = focalLength_m;

  // clang-format off
  float m[16];
  m[ 0] = pixelSizeX; m[ 4] = 0.0f;        m[ 8] = 0.0f; m[12] = tx;
  m[ 1] = 0.0f;       m[ 5] = -pixelSizeY; m[ 9] = 0.0f; m[13] = ty;
  m[ 2] = 0.0f;       m[ 6] = 0.0f;        m[10] = 1.0f; m[14] = tz;
  m[ 3] = 0.0f;       m[ 7] = 0.0f;        m[11] = 0.0f; m[15] = 1.0f;
  // clang-format on

  return Transform{m};
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
