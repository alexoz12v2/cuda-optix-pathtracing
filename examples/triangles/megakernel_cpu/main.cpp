#include "cuda-core/types.cuh"
#include "cuda-core/bsdf.cuh"
#include "cuda-core/rng.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/extra_math.cuh"
#include "cuda-core/host_scene.cuh"
#include "cuda-core/host_utils.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/morton.cuh"
#include "cuda-core/shapes.cuh"
#include "cuda-core/kernels.cuh"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

namespace {

Ray cameraToPixelCenterRay(int2 pixel, Transform const& cameraFromRaster,
                           Transform const& renderFromCamera) {
  Ray ray{};

  const auto [rasterX, rasterY] = make_float2(pixel.x + 0.5f, pixel.y + 0.5f);
  float3 const pCamera =
      cameraFromRaster.apply(make_float3(rasterX, rasterY, 0.f));

  ray.o = renderFromCamera.apply(make_float3(0.0f, 0.0f, 0.0f));
  ray.d = normalize(renderFromCamera.applyDirection(pCamera));

  return ray;
}

void hostIntersectCore(int mortonOrNegative, int2 pixel,
                       Transform const& cameraFromRaster,
                       Transform const& renderFromCamera,
                       std::vector<Triangle> const& triangles,
                       std::function<void(int2, int, float)> const& onHit) {
  Ray const ray =
      cameraToPixelCenterRay(pixel, cameraFromRaster, renderFromCamera);
  HitResult hitResult{};
  hitResult.t = std::numeric_limits<float>::infinity();
  for (Triangle const& tri : triangles) {
    HitResult const result =
        hostIntersectMT(ray.o, ray.d, tri.v0, tri.v1, tri.v2);
    if (result.hit && result.t < hitResult.t) {
      hitResult = result;
    }
  }
  if (hitResult.hit) {
    onHit(pixel, mortonOrNegative, hitResult.t);
  }
}

void hostIntersectionKernel(
    bool morton, DeviceCamera cam, std::vector<Triangle> const& triangles,
    std::function<void(int2, int, float)> const& onHit) {
  MortonLayout2D const layout = mortonLayout(cam.height, cam.width);

  Transform const cameraFromRaster = cameraFromRaster_Perspective(
      cam.focalLength, cam.sensorSize, cam.width, cam.height);
  Transform const renderFromCamera = worldFromCamera(cam.dir, cam.pos);

  if (morton) {
    for (int i = 0; i < layout.mortonCount; ++i) {
      uint2 upixel{};
      decodeMorton2D(i, &upixel.x, &upixel.y);
      if (upixel.x >= cam.width || upixel.y >= cam.height) {
        continue;
      }

      int2 const pixel = make_int2(upixel.x, upixel.y);
      hostIntersectCore(i, pixel, cameraFromRaster, renderFromCamera, triangles,
                        onHit);
    }
  } else {
    for (int row = 0; row < cam.height; ++row) {
      for (int col = 0; col < cam.width; ++col) {
        int2 const pixel = make_int2(col, row);
        // std::cout << "PIXEL " << col << " " << row << std::endl;
        hostIntersectCore(-1, pixel, cameraFromRaster, renderFromCamera,
                          triangles, onHit);
      }
    }
  }
}

void megakernelMain() {
  std::cout << "Initializing the scene" << std::endl;
  HostTriangleScene h_scene;
  std::vector<Light> h_lights;
  std::vector<Light> h_infiniteLights;
  std::vector<BSDF> h_bsdfs;
  DeviceCamera h_camera;
  cornellBox(&h_scene, &h_lights, &h_infiniteLights, &h_bsdfs, &h_camera);
  std::cout << "Computing intersection to host" << std::endl;
  std::vector<float4> out;
  out.resize(mortonLayout(h_camera.height, h_camera.width).mortonCount);
  hostIntersectionKernel(true, h_camera, h_mesh,
                         [&](int2 pixel, int mortonIdx, float t) {
                           float const value = std::isfinite(t) ? t / 2 : 0;
                           out[mortonIdx] = make_float4(.5f, .5f, value, 1.f);
                         });
  std::cout << "Writing to host" << std::endl;
  writeOutputBuffer(out.data(), h_camera.width, h_camera.height,
                    "output-host.bmp", true);
}

}  // namespace

// UNICODE and _UNICODE always defined
#ifdef _WIN32
int wmain() {
#else
int main() {
#endif
#ifdef DMT_OS_WINDOWS
  SetConsoleOutputCP(CP_UTF8);
  for (DWORD conoutHandleId : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
    HANDLE const hConsole = GetStdHandle(conoutHandleId);
    DWORD mode = 0;
    if (GetConsoleMode(hConsole, &mode)) {
      mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    }
  }
#endif
  megakernelMain();
}
