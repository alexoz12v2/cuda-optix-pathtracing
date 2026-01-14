#include "megakernel.cuh"

#include "cuda-core/debug.cuh"
#include "cuda-core/types.cuh"
#include "cuda-core/morton.cuh"
#include "cuda-core/extra_math.cuh"

#include <cooperative_groups.h>

#include <numbers>

namespace cg = cooperative_groups;

__constant__ float CMEM_cameraFromRaster[32];
__constant__ float CMEM_renderFromCamera[32];
__constant__ DeviceHaltonOwenParams CMEM_haltonOwenParams;
__constant__ int2 CMEM_imageResolution;
__constant__ int CMEM_spp;
__constant__ MortonLayout2D CMEM_mortonLayout;

__host__ void allocateDeviceConstantMemory(DeviceCamera const& h_camera) {
  {
    Transform t =
        cameraFromRaster_Perspective(h_camera.focalLength, h_camera.sensorSize,
                                     h_camera.width, h_camera.height);
    CUDA_CHECK(cudaMemcpyToSymbol(CMEM_cameraFromRaster, &t, sizeof(Transform),
                                  0, cudaMemcpyHostToDevice));
    t = worldFromCamera(h_camera.dir, h_camera.pos);
    CUDA_CHECK(cudaMemcpyToSymbol(CMEM_renderFromCamera, &t, sizeof(Transform),
                                  0, cudaMemcpyHostToDevice));
  }
  int2 const res = make_int2(h_camera.width, h_camera.height);
  CUDA_CHECK(cudaMemcpyToSymbol(CMEM_imageResolution, &res, sizeof(int2), 0,
                                cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(CMEM_spp, &h_camera.spp, sizeof(int), 0,
                                cudaMemcpyHostToDevice));
  {
    DeviceHaltonOwenParams const rngParams =
        DeviceHaltonOwen::computeParams(h_camera.width, h_camera.height);
    CUDA_CHECK(cudaMemcpyToSymbol(CMEM_haltonOwenParams, &rngParams,
                                  sizeof(DeviceHaltonOwenParams), 0,
                                  cudaMemcpyHostToDevice));
  }
  {
    MortonLayout2D const morton = mortonLayout(h_camera.width, h_camera.height);
    CUDA_CHECK(cudaMemcpyToSymbol(CMEM_mortonLayout, &morton,
                                  sizeof(MortonLayout2D), 0,
                                  cudaMemcpyHostToDevice));
  }
  allocateDeviceGGXEnergyPreservingTables();
}

__global__ void pathTraceMegakernel(
    DeviceCamera* d__cam, TriangleSoup d_triSoup, Light const* d_lights,
    uint32_t const lightCount, Light const* d_infiniteLights,
    uint32_t const infiniteLightCount, BSDF const* d_bsdf,
    uint32_t const bsdfCount, uint32_t const sampleOffset,
    DeviceHaltonOwen* d_haltonOwen,
#if DMT_ENABLE_MSE
    DeviceOutputBuffer d_out
#else
    float4* d_outBuffer
#endif
) {
  // grid-stride loop start
  uint32_t const mortonStart = blockIdx.x * blockDim.x + threadIdx.x;

  // fetch rng
  DeviceHaltonOwen& warpRng = d_haltonOwen[mortonStart / warpSize];

  // Declare extern SMEM
#if DMT_ENABLE_MSE
  auto& SMEM = SMEMLayout<64>::get();
#endif

  // constant memory aliases
  uint32_t const& spp = CMEM_spp;
  MortonLayout2D const& layout = CMEM_mortonLayout;
  DeviceHaltonOwenParams const& params = CMEM_haltonOwenParams;
  Transform const& cameraFromRaster = *arrayAsTransform(CMEM_cameraFromRaster);
  Transform const& renderFromCamera = *arrayAsTransform(CMEM_renderFromCamera);
  int2 const& imageRes = CMEM_imageResolution;

  for (uint32_t mortonIndex = mortonStart; mortonIndex < layout.mortonCount;
       mortonIndex += gridDim.x * blockDim.x) {
    uint2 upixel{};
    decodeMorton2D(mortonIndex, &upixel.x, &upixel.y);
    if (upixel.x >= imageRes.x || upixel.y >= imageRes.y) {
      continue;
    }
    int2 const pixel = make_int2(upixel.x, upixel.y);
    float4* meanPtr = d_out.meanPtr + pixel.x + pixel.y * imageRes.x;
    float4* M2Ptr = d_out.m2Ptr + pixel.x + pixel.y * imageRes.x;

#define OFFSET_RAY_ORIGIN 1
#define FIRST_BOUNCE_ONLY 0

#if DMT_ENABLE_MSE
    SMEM.startSample(meanPtr, M2Ptr);
#endif

    for (int sbase = 0; sbase < spp; ++sbase) {
      int const s = sbase + sampleOffset;
      warpRng.startPixelSample(params, pixel, s);
      PRINT("Starting sample px: %d %d s: %d\n", pixel.x, pixel.y, s);

      BSDF bsdf{};
      int depth = 0;
      int transmissionCount = 0;
      bool specularBounce = false;
      bool lastBounceTransmission = false;
      float3 L = make_float3(0, 0, 0);
      float3 beta = make_float3(1, 1, 1);
      cg::coalesced_group const theWarp = cg::coalesced_threads();
      Ray ray = getCameraRay(getCameraSample(pixel, warpRng, params),
                             cameraFromRaster, renderFromCamera);
      while (true) {
        PRINT("  [%d %d] Ray Depth %d\n", pixel.x, pixel.y, depth);
        // intersect with scene
        HitResult hitResult;
        for (int tri = 0; tri < d_triSoup.count; ++tri) {
          float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
          float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
          float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
          HitResult const result = triangleIntersect(x, y, z, ray);
          if (result.hit && result.t < hitResult.t) {
            hitResult = result;
            hitResult.matId = d_triSoup.matId[tri];
            if (dot(ray.d, hitResult.normal) > 0) {
              hitResult.normal *= -1;
            }
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
          PRINT("   [%d %d:%d] No Intersection. Killing path\n", pixel.x,
                pixel.y, depth);
          break;
        }

        // TODO
        static int constexpr MAX_DEPTH = 32;
        if (depth >= MAX_DEPTH) {
          PRINT("Max depth reached\n");
          break;
        }

        // update last bounce BSDF
        PRINT("    [%d %d:%d] Intersection at %f %f %f with normal %f %f %f.\n",
              pixel.x, pixel.y, depth, hitResult.pos.x, hitResult.pos.y,
              hitResult.pos.z, hitResult.normal.x, hitResult.normal.y,
              hitResult.normal.z);
        bsdf = d_bsdf[hitResult.matId];
        prepareBSDF(&bsdf, hitResult.normal, -ray.d, transmissionCount);
        PRINT("    - [%d %d:%d] Prepared BSDF.\n", pixel.x, pixel.y, depth);

        // - choose light for Next Event Estimation (direct lighting)
        int32_t const lightIndex =
            min(static_cast<int>(warpRng.get1D(params) * lightCount),
                lightCount - 1);
        Light const light = d_lights[lightIndex];
        float const lightPMF = 1.f / lightCount;
        PRINT("    - [%d %d:%d] Fetched Light.\n", pixel.x, pixel.y, depth);
        if (LightSample const ls =
                sampleLight(light, hitResult.pos, warpRng.get2D(params),
                            lastBounceTransmission, hitResult.normal);
            ls) {
          PRINT("    [%d %d:%d] Sampled Light. Evaluating shadow ray.\n",
                pixel.x, pixel.y, depth);
          // - create shadow ray (offset origin to avoid self intersection)
#if DMT_ENABLE_ASSERTS
          if (float3 diff = normalize(ls.pLight - hitResult.pos);
              !componentWiseNear(diff, ls.direction)) {
            printf(
                "b: [%u %u %u] t: [%u %u %u] px: [%d %d] d: %d\n\t"
                "diff: %f %f %f\n\tlsdi: %f %f %f\n",
                blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
                threadIdx.z, pixel.x, pixel.y, depth, diff.x, diff.y, diff.z,
                ls.direction.x, ls.direction.y, ls.direction.z);
          }
          assert(componentWiseNear(normalize(ls.pLight - hitResult.pos),
                                   ls.direction));
#endif
          Ray const shadowRay{
#if OFFSET_RAY_ORIGIN
              offsetRayOrigin(hitResult.pos, hitResult.error, hitResult.normal,
                              ls.direction),
#else
              hitResult.pos,
#endif
              ls.direction};
          // - trace ray (TODO flip normal on odd tranmission count)
          bool doNextEventEstimation = true;
          for (int tri = 0; tri < d_triSoup.count; ++tri) {
            float4 const x = reinterpret_cast<float4 const*>(d_triSoup.xs)[tri];
            float4 const y = reinterpret_cast<float4 const*>(d_triSoup.ys)[tri];
            float4 const z = reinterpret_cast<float4 const*>(d_triSoup.zs)[tri];
            if (HitResult const result = triangleIntersect(x, y, z, shadowRay);
                result.hit && result.t < ls.distance) {
              PRINT("    [%d %d:%d] Shadow Occlusion with %d. Skip NEE.\n",
                    pixel.x, pixel.y, depth, tri);
              doNextEventEstimation = false;
              break;
            }
          }
          // - if no occlusion, sample light and add light contribution
          if (doNextEventEstimation) {
            PRINT("    [%d %d:%d] No Occlusion. Performing NEE.\n", pixel.x,
                  pixel.y, depth);
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
                float const w =
                    sqrf(lightPMF * ls.pdf) / sqrf(lightPMF * ls.pdf + bsdfPdf);
                L += Le * bsdf_f * beta * w;
              }
            }
          }
        }
#if FIRST_BOUNCE_ONLY
        break;
#endif

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
        lastBounceTransmission = bs.refract;
        // next ray
#if DMT_ENABLE_ASSERTS
        if (!bs.refract && dot(bs.wi, hitResult.normal) <= 0) {
          printf(
              "%d %d %d refract: %d, wi: %f %f %f, n: %f %f %f\n"
              "       pos: %f %f %f\n",
              pixel.x, pixel.y, depth, bs.refract, bs.wi.x, bs.wi.y, bs.wi.z,
              hitResult.normal.x, hitResult.normal.y, hitResult.normal.z,
              hitResult.pos.x, hitResult.pos.y, hitResult.pos.z);
        }
        assert(bs.refract || dot(bs.wi, hitResult.normal) > 0);
#endif
#if OFFSET_RAY_ORIGIN
        ray.o = offsetRayOrigin(hitResult.pos, hitResult.error,
                                hitResult.normal, bs.wi);
#else
        ray.o = hitResult.pos;
#endif
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

      // add to output buffer: Read-Modify-Update procedure
#if DMT_ENABLE_MSE
      SMEM.updateSample(L);
#else
      float3 mean = make_float3(meanPtr->x, meanPtr->y, meanPtr->z);
      float N = meanPtr->w;

      N += 1.0f;
      float3 delta = L - mean;
      mean += delta / N;

      // write back
      meanPtr->x = mean.x;
      meanPtr->y = mean.y;
      meanPtr->z = mean.z;
      meanPtr->w = N;
#endif
    }

#if DMT_ENABLE_MSE
    SMEM.endSample(meanPtr, M2Ptr);
#endif
  }
}
