#include <math_constants.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "common.cuh"

namespace cg = cooperative_groups;

__device__ CameraSample getCameraSample(int2 pPixel, DeviceHaltonOwen& rng,
                                        DeviceHaltonOwenParams const& params) {
  CameraSample cs{};

  // TODO filter result from getPixel2D
  float2 const pixelShift = rng.getPixel2D(params) - make_float2(0.5f, 0.5f);

  cs.pFilm =
      pixelShift + make_float2(0.5, 0.5) + make_float2(pPixel.x, pPixel.y);
  cs.filterWeight = 1.f;  // TODO filtering
  return cs;
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
// test kernel
__global__ void raygenKernel(DeviceCamera* d_cam,
                             DeviceHaltonOwen* d_haltonOwen, RayTile* d_rays) {
  // transforms
  // float3 pointCam();
  //  grid-style loop (2D)
  //  - getCameraRay
  //  - write to buffer
}

// 128-byte boundary aligned staring address
// 0x____'____'____'____'____'____'____'__00
// 0x____'____'____'____'____'____'____'__80

// 32-byte boundary aligned staring address
// 0x____'____'____'____'____'____'____'__00
// 0x____'____'____'____'____'____'____'__20
// 0x____'____'____'____'____'____'____'__40
// 0x____'____'____'____'____'____'____'__60
// 0x____'____'____'____'____'____'____'__80
// 0x____'____'____'____'____'____'____'__a0
// 0x____'____'____'____'____'____'____'__c0
// 0x____'____'____'____'____'____'____'__e0

__host__ __device__ Transform worldFromCamera(float3 cameraDirection,
                                              float3 cameraPosition) {
  float3 const cameraRight =
      normalize(cross(make_float3(0, 0, 1), cameraDirection));  // X
  float3 const cameraUp = make_float3(0, 0, 1);                 // Y → world Z
  float3 const cameraForward = normalize(cameraDirection);      // Z → world Y

  // Column-major matrix for worldFromCamera
  // clang-format off
  float m[16];
  m[ 0] = cameraRight.x; m[ 4] = cameraUp.x; m[ 8] = cameraForward.x; m[12] = cameraPosition.x;
  m[ 1] = cameraRight.y; m[ 5] = cameraUp.y; m[ 9] = cameraForward.y; m[13] = cameraPosition.y;
  m[ 2] = cameraRight.z; m[ 6] = cameraUp.z; m[10] = cameraForward.z; m[14] = cameraPosition.z;
  m[ 3] = 0.0f;          m[ 7] = 0.0f;       m[11] = 0.0f;            m[15] = 1.0f;
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

__global__ void basicIntersectionMegakernel(DeviceCamera* d_cam,
                                            TriangleSoup d_triSoup,
                                            DeviceHaltonOwen* d_haltonOwen,
                                            float* d_outBuffer) {
  uint32_t const mortonIndex = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const spp = d_cam->spp;
  // constant memory?
  MortonLayout2D const layout = mortonLayout(d_cam->width, d_cam->height);
  DeviceHaltonOwenParams const params =
      d_haltonOwen[mortonIndex / warpSize].computeParams(layout.cols,
                                                         layout.rows);
  Transform const cameraFromRaster = cameraFromRaster_Perspective(
      d_cam->focalLength, d_cam->sensorSize, d_cam->width, d_cam->height);
  Transform const renderFromCamera = worldFromCamera(d_cam->dir, d_cam->pos);

  for (int i = mortonIndex; i < layout.mortonCount;
       i += gridDim.x * blockDim.x) {
    uint2 upixel{};
    decodeMorton2D(i, &upixel.x, &upixel.y);
    int2 const pixel = make_int2(upixel.x, upixel.y);
    for (int s = 0; s < spp; ++s) {
      // start pixel sample and generate ray
      d_haltonOwen->startPixelSample(params, pixel, s);
      Ray const ray =
          getCameraRay(getCameraSample(pixel, *d_haltonOwen, params),
                       cameraFromRaster, renderFromCamera);
      // intersect with scene
      HitResult hitResult{};
      hitResult.t = CUDART_INF_F;
      for (int tri = 0; tri < d_triSoup.count; ++tri) {
        float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
        float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
        float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
        HitResult const result = triangleIntersect(x, y, z, ray);
        if (result.hit && result.t < hitResult.t) {
          hitResult = result;
        }
      }
      __syncwarp();

      // add to output buffer
      float const toAdd = static_cast<float>(hitResult.hit) / spp;
      atomicAdd(&d_outBuffer[mortonIndex], toAdd);
    }
  }
}