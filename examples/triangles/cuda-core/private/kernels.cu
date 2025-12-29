#include "kernels.cuh"

#include "debug.cuh"
#include "types.cuh"
#include "morton.cuh"
#include "extra_math.cuh"

#include <cooperative_groups.h>

#include <numbers>

namespace cg = cooperative_groups;

__global__ void __launch_bounds__(/*max threads per block*/ 512,
                                  /*min blocks per SM*/ 10)
    basicIntersectionMegakernel(
        DeviceCamera* d_cam, TriangleSoup d_triSoup, Light const* d_lights,
        uint32_t const lightCount, Light const* d_infiniteLights,
        uint32_t const infiniteLightCount, BSDF const* d_bsdf,
        uint32_t const bsdfCount, uint32_t const sampleOffset,
        DeviceHaltonOwen* d_haltonOwen, float4* d_outBuffer) {
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
  for (uint32_t mortonIndex = mortonStart; mortonIndex < layout.mortonCount;
       mortonIndex += gridDim.x * blockDim.x) {
    uint2 upixel{};
    decodeMorton2D(mortonIndex, &upixel.x, &upixel.y);
    if (upixel.x >= d_cam->width || upixel.y >= d_cam->height) {
      continue;
    }
    int2 const pixel = make_int2(upixel.x, upixel.y);

    for (int sbase = 0; sbase < spp; ++sbase) {
      int const s = sbase + sampleOffset;
      warpRng.startPixelSample(params, pixel, s);
      PRINT("Starting sample px: %d %d s: %d\n", pixel.x, pixel.y, s);

      BSDF bsdf{};
      int depth = 0;
      int transmissionCount = 0;
      bool specularBounce = false;
      float3 L = make_float3(0, 0, 0);
      float3 beta = make_float3(1, 1, 1);
      cg::coalesced_group const theWarp = cg::coalesced_threads();
      Ray ray = getCameraRay(getCameraSample(pixel, warpRng, params),
                             cameraFromRaster, renderFromCamera);
      while (true) {
        PRINT("  Ray Depth %d\n", depth);
        // intersect with scene
        HitResult hitResult;
        for (int tri = 0; tri < d_triSoup.count; ++tri) {
          float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
          float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
          float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
          HitResult const result = triangleIntersect(x, y, z, ray);
          if (result.hit && result.t < hitResult.t) {
            hitResult = result;
            // TODO better? Coalescing?
            hitResult.matId = d_triSoup.matId[tri];
          }
        }
        // if not intersected, contribution is zero (eliminates branch later)
        if (!hitResult.hit) {
          int32_t const lightIndex =
              min(static_cast<int>(warpRng.get1D(params) * infiniteLightCount),
                  infiniteLightCount - 1);
          Light const light = d_infiniteLights[lightIndex];
          float const lightPMF = 1.f / infiniteLightCount;
          float pdf = 0;
          float3 const Le = evalInfiniteLight(light, ray.d, &pdf);
          if (pdf) {
            // TODO: If any specular bounce or depth == 0, don't do MIS
            // TODO: Use last BSDF for MIS
            L += beta * Le / lightPMF;
          }
          PRINT("    No Intersection. Killing path\n");
          break;
        }

        static int constexpr MAX_DEPTH = 32;
        if (depth >= MAX_DEPTH) {
          PRINT("Max depth reached\n");
          break;
        }

        // update last bounce BSDF
        PRINT("    Intersection at %f %f %f with normal %f %f %f.\n",
              hitResult.pos.x, hitResult.pos.y, hitResult.pos.z,
              hitResult.normal.x, hitResult.normal.y, hitResult.normal.z);
        bsdf = d_bsdf[hitResult.matId];
        prepareBSDF(&bsdf, hitResult.normal, -ray.d);
        PRINT("    - Prepared BSDF.\n");

        // - choose light for Next Event Estimation (direct lighting)
        int32_t const lightIndex =
            min(static_cast<int>(warpRng.get1D(params) * lightCount),
                lightCount - 1);
        Light const light = d_lights[lightIndex];
        float const lightPMF = 1.f / lightCount;
        PRINT("    - Fetched Light.\n");
        if (LightSample const ls =
                sampleLight(light, hitResult.pos, warpRng.get2D(params), false,
                            hitResult.normal);
            ls) {
          PRINT("    Sampled Light. Evaluating shadow ray.\n");
          // - create shadow ray (offset origin to avoid self intersection)
          Ray const shadowRay{offsetRayOrigin(hitResult.pos, hitResult.error,
                                              hitResult.normal, ls.direction),
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
          // - if no occlusion, sample light and add light contribution
          if (doNextEventEstimation) {
            float bsdfPdf = 0;
            float3 const bsdf_f =
                evalBsdf(bsdf, -ray.d, shadowRay.d, hitResult.normal,
                         hitResult.normal, &bsdfPdf) *
                bsdf.weight();
            // bsdfDelta always false TODO assert
            float3 const Le = evalLight(light, ls);
            if (!isZero(bsdf_f)) {
              if (ls.delta) {
                L += beta * Le * bsdf_f / lightPMF;
              } else {
                // MIS if not delta (and not BSDF Delta TODO)
                // power heuristic
                float const w = sqrf(10 * lightPMF * ls.pdf) /
                                sqrf(10 * lightPMF * ls.pdf + bsdfPdf);
                L += Le * bsdf_f * beta * w;
              }
            }
          }
          PRINT("    No Occlusion. Performing NEE.\n");
        }

        // Bounce
        BSDFSample const bs =
            sampleBsdf(bsdf, -ray.d, hitResult.normal, hitResult.normal,
                       warpRng.get2D(params), warpRng.get1D(params));
        PRINT("    Sampling BSDF for next bounce.\n");
        if (!bs) {
          PRINT("    Sampling BSDF Failed. Killing path.\n");
          break;
        }
        PRINT("    Sampled BSDF: delta: %d refract: %d. eta: %f\n", bs.delta,
              bs.refract, bs.eta);
        // update state
        specularBounce |= bs.delta;
        transmissionCount += bs.refract;
        // next ray (TODO offset ray origin)
        assert(bs.refract || dot(bs.wi, hitResult.normal) > 0);
        ray.o = offsetRayOrigin(hitResult.pos, hitResult.error,
                                hitResult.normal, bs.wi);
        ray.d = bs.wi;
        // update path throughput
        beta *= bs.f * fabsf(dot(bs.wi, hitResult.normal)) / bs.pdf;

        // russian roulette and ray depth
        if (float const rrBeta = maxComponentValue(beta * bs.eta);
            rrBeta < 1 && depth > 1) {
          PRINT("    Rolling Russian Roulette with %f\n", rrBeta);
          float const q = fmaxf(0.f, 1.f - rrBeta);
          if (warpRng.get1D(params) < q) {
            PRINT("    Path Killed Through Russian Roulette\n");
            break;
          }
          PRINT("    Path Survived, amplifying throughput\n");
          beta /= 1 - q;
        }
        ++depth;
      }  // end while
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
    }
  }
}
