#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH

#include "cudautils/cudautils-macro.cuh"
#include "cudautils/cudautils-transform.cuh"
#include "cudautils/cudautils-vecmath.cuh"

namespace dmt {
    struct DeviceCamera
    {
        float    focalLength = 20.f;
        float    sensorSize  = 36.f;
        Vector3f dir{0.f, 1.f, 0.f};
        //float camDirX     = 0.f;
        //float camDirY     = 1;
        //float camDirZ     = 0.f;
        int spp = 4;
        //float camPosX     = 0.f;
        //float camPosY     = 0.f;
        //float camPosZ     = 0.f;
        Vector3f pos{0.f, 0.f, 0.f};
        int      width  = 128;
        int      height = 128;
    };

    DMT_GPU void generate_camera_ray(DeviceCamera cam, int px, int py, float u1, float u2, float3& ro, float3& rd);

} // namespace dmt
#endif // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH
