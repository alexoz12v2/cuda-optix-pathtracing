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
  ray.d = normalize(renderFromCamera.apply(normalize(pCamera)));

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

// | Rx Ux Fx Tx |
// | Ry Uy Fy Ty |
// | Rz Uz Fz Tz |
// | 0  0  0  1  |
__device__ Transform worldFromCamera(float3 cameraDirection,
                                     float3 cameraPosition) {
  float3 const forward = cameraDirection;  // +Z in camera space
  float3 const worldUp{0.0f, 0.0f, 1.0f};  // World up (0, 0, +Z)

  // Compute right (X) and up (Y) vectors for the camera frame (left handed
  // system)
  float3 const right =
      normalize(cross(forward, worldUp));              // +X in camera space
  float3 const up = normalize(cross(right, forward));  // +Y in camera space

  // Column-major matrix for worldFromCamera
  // clang-format off
  float const m[16]{
    right.x,   right.y,   right.z,   0.0f, // Column 0: right
    up.x,      up.y,      up.z,      0.0f, // Column 1: up
    forward.x, forward.y, forward.z, 0.0f, // Column 2: forward
    cameraPosition.x, cameraPosition.y, cameraPosition.z, 1.0f // Column 3: position
  };
  // clang-format on
  return Transform{m};
}

__device__ Transform cameraFromRaster_Perspective(float focalLength,
                                                  float sensorHeight,
                                                  uint32_t xRes,
                                                  uint32_t yRes) {
  float const aspectRatio = static_cast<float>(xRes) / static_cast<float>(yRes);
  float const halfHeight = 0.5f * sensorHeight;
  float const halfWidth = halfHeight * aspectRatio;

  float const pixelSizeX = 2.0f * halfWidth / static_cast<float>(xRes);
  float const pixelSizeY = 2.0f * halfHeight / static_cast<float>(yRes);

  float const tx = -halfWidth + 0.5f * pixelSizeX;
  float const ty = halfHeight - 0.5f * pixelSizeY;

  float const m[16]{0.0f, 0.0f, 1.0f, 0.0f, tx, ty, focalLength, 1.0f};

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

      // add to output buffer
      float const toAdd = static_cast<float>(hitResult.hit) / spp;
      atomicAdd(&d_outBuffer[mortonIndex], toAdd);
    }
  }
}