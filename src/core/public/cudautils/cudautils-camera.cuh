#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH

#include "core/cudautils/cudautils-macro.cuh"
#include <core/cudautils/cudautils-spectrum.cuh>
#include <core/cudautils/cudautils-transform.cuh>
#include <core/cudautils/cudautils-vecmath.cuh>
#include <core/cudautils/cudautils-film.cuh>

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

    DMT_GPU void generate_camera_ray(
        DeviceCamera cam,
        int          px,
        int          py,
        float        u1,
        float        u2,
        float&       rox,
        float&       roy,
        float&       roz,
        float&       rdx,
        float&       rdy,
        float&       rdz);

} // namespace dmt
#endif // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_CAMERA_CUH
