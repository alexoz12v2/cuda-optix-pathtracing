#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH

#include "cudautils/cudautils-macro.cuh"
#include "cudautils/cudautils-transform.cuh"
#include "cudautils/cudautils-vecmath.cuh"

namespace dmt {
struct DeviceCamera {
  float focalLength = 20.f;
  float sensorSize = 36.f;
  Vector3f dir{0.f, 1.f, 0.f};
  int spp = 4;
  Vector3f pos{0.f, 0.f, 0.f};
  int width = 128;
  int height = 128;
};

DMT_GPU void generate_camera_ray(DeviceCamera cam, int px, int py, float u1,
                                 float u2, float3& ro, float3& rd);

DMT_GPU void generate_camera_ray_mega(DeviceCamera cam, int px, int py,
                                      float u1, float u2, float3& ro,
                                      float3& rd);

// Ray generateRay(CameraSample const& cs, Transform const& cameraFromRaster,
// Transform const& renderFromCamera)
//{
//     Point3f const pxImage{{cs.pFilm.x, cs.pFilm.y, 0}};
//     Point3f const pCamera{{cameraFromRaster(pxImage)}};
//     // TODO add lens?
//     // time should use lerp(cs.time, shutterOpen, shutterClose)
//     Ray const ray{Point3f{{0, 0, 0}}, normalize(pCamera), cs.time};
//     // TODO handle tMax better
//     float tMax = 1e5f;
//     return renderFromCamera(ray, &tMax);
// }
//  struct PRay
//  {
//      Vector3f;
//  };
}  // namespace dmt
#endif  // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH
