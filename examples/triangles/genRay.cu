#include <math_constants.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "common.cuh"

#include <numbers>

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

__global__ void __launch_bounds__(/*max threads per block*/ 512,
                                  /*min blocks per SM*/ 10)
    basicIntersectionMegakernel(DeviceCamera* d_cam, TriangleSoup d_triSoup,
                                Light const* d_lights,
                                uint32_t const lightCount,
                                Light const* d_infiniteLights,
                                uint32_t const infiniteLightCount,
                                DeviceHaltonOwen* d_haltonOwen,
                                float4* d_outBuffer) {
  uint32_t const mortonStart = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const spp = d_cam->spp;
  // constant memory?
  MortonLayout2D const layout = mortonLayout(d_cam->width, d_cam->height);
  DeviceHaltonOwen& warpRng = d_haltonOwen[mortonStart / warpSize];
  DeviceHaltonOwenParams const params =
      warpRng.computeParams(layout.cols, layout.rows);
  Transform const cameraFromRaster = cameraFromRaster_Perspective(
      d_cam->focalLength, d_cam->sensorSize, d_cam->width, d_cam->height);
  Transform const renderFromCamera = worldFromCamera(d_cam->dir, d_cam->pos);
  // #define PRINT(...) printf(__VA_ARGS__)
#define PRINT(...)
#if 0
  printf("-------------- Triangles ---------------\n");
  for (int tri = 0; tri < d_triSoup.count; ++tri) {
    float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
    float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
    float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
    printf("[%d]: \n\t%f %f %f\n\t%f %f %f\n\t %f %f %f\n", tri, x.x, y.x, z.x,
           x.y, y.y, z.y, x.z, y.z, z.z);
  }
#endif
  // TODO: skip morton indices which are cut
  for (int mortonIndex = mortonStart; mortonIndex < layout.mortonCount;
       mortonIndex += gridDim.x * blockDim.x) {
    uint2 upixel{};
    decodeMorton2D(mortonIndex, &upixel.x, &upixel.y);
    int2 const pixel = make_int2(upixel.x, upixel.y);
#define DO_SAMPLING 1
#if !DO_SAMPLING
    {
      PRINT("Start pixel %d %d\n", pixel.x, pixel.y);
      HitResult hitResult{};
      hitResult.t = CUDART_INF_F;
      Ray const ray =
          getCameraRayNonRandom(pixel, cameraFromRaster, renderFromCamera);
      PRINT("----------------------------------------------------\n");
      PRINT("Ray: \n");
      PRINT("    Origin: %f %f %f \n", ray.o.x, ray.o.y, ray.o.z);
      PRINT("    Direct: %f %f %f \n", ray.d.x, ray.d.y, ray.d.z);
      for (int tri = 0; tri < d_triSoup.count; ++tri) {
        float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
        float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
        float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
        HitResult const result = triangleIntersect(x, y, z, ray);
        if (result.hit && result.t < hitResult.t) {
          hitResult = result;
        }
      }
      // TODO remove
      if (hitResult.hit) {
        PRINT("    Intersection at t = %f\n", hitResult.t);
      }
      PRINT("----------------------------------------------------\n");
#else
    PRINT("----------------------------------------------------\n");
    PRINT("----------------------------------------------------\n");
    for (int s = 0; s < spp; ++s) {
      // TODO refactor into path state?
      float3 L = make_float3(0, 0, 0);
      float beta = 1.f;
      cg::coalesced_group const theWarp = cg::coalesced_threads();

      PRINT("----------------------------------------------------\n");
      PRINT("Start pixel %d %d | Sample %d\n", pixel.x, pixel.y, s);
      // start pixel sample and generate ray
      warpRng.startPixelSample(params, pixel, s);
      Ray const ray = getCameraRay(getCameraSample(pixel, warpRng, params),
                                   cameraFromRaster, renderFromCamera);
      PRINT("Ray: \n");
      PRINT("    Origin: %f %f %f \n", ray.o.x, ray.o.y, ray.o.z);
      PRINT("    Direct: %f %f %f \n", ray.d.x, ray.d.y, ray.d.z);
      PRINT("----------------------------------------------------\n");
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
      // TODO remove
      if (hitResult.hit) {
        PRINT("    Intersection at t = %f\n", hitResult.t);
      }
#endif
      // TODO probably remove
      // if not intersected, contribution is zero (eliminates branch later)
      if (!hitResult.hit) {
        // TODO all env lights here
        int32_t const lightIndex =
            min(static_cast<int>(warpRng.get1D(params) * infiniteLightCount),
                infiniteLightCount - 1);
        Light const light = d_infiniteLights[lightIndex];
        float const lightPMF = 1 / infiniteLightCount;
        float pdf = 0;
        float3 const Le = evalInfiniteLight(light, ray.d, &pdf);
        if (pdf) {
          // TODO: If any specular bounce or depth == 0, don't do MIS
          // TODO: Use last BSDF for MIS
          L += beta * Le;
        }
      } else {
        // - choose light for Next Event Estimation (direct lighting)
        int32_t const lightIndex =
            min(static_cast<int>(warpRng.get1D(params) * lightCount),
                lightCount - 1);
        Light const light = d_lights[lightIndex];
        float const lightPMF = 1 / lightCount;
        if (LightSample const ls =
                sampleLight(light, hitResult.pos, warpRng.get2D(params), false,
                            hitResult.normal);
            ls) {
          // - create shadow ray (offset origin to avoid self intersection)
          Ray const shadowRay{
              length(hitResult.error) * hitResult.normal + hitResult.pos,
              ls.direction};
          // - trace ray
          bool doNextEventEstimation = true;
          for (int tri = 0; tri < d_triSoup.count; ++tri) {
            float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
            float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
            float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
            if (HitResult const result = triangleIntersect(x, y, z, shadowRay);
                result.hit) {
              doNextEventEstimation = false;
              break;
            }
          }
          // - if no intersection, sample light and add light contribution
          if (doNextEventEstimation) {
            // TODO attenuation and bsdf factor evaluation
            float constexpr bsdfFactor = 1.f;
            float constexpr bsdfPdf = 0.25 * std::numbers::pi_v<float>;
            float3 const Le = evalLight(light, ls);
            // TODO revise
            if (ls.delta) {
              // MIS if not delta (and not BSDF Delta TODO)
              L += beta * Le * bsdfFactor / lightPMF;
            } else {
              // otherwise (to check bsdf not delta?? TODO)
              L += beta * (Le * bsdfFactor / (lightPMF * ls.pdf + bsdfPdf));
            }
          }
        }
      }
      theWarp.sync();

      // add to output buffer ( assumes max distance is 2)
      // TODO the rest (this is a branch BTW)
      float4* target = d_outBuffer + mortonIndex;
#if 1  // each pixel now is associated to a thread, no atomic
      // TODO: better coalescing?
      target->x += L.x;
      target->y += L.y;
      target->z += L.z;
      target->w += 1;  // TODO: sum of weights
#else
      atomicAdd(target, toAdd);
#endif

      // TODO russian roulette
    }
  }
}