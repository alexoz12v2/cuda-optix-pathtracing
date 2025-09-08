#pragma once

#include "cudautils/cudautils-macro.h"
#include <cudautils/cudautils.h>
#include <cudautils/cudautils-spectrum.h>
#include <cudautils/cudautils-transform.h>
#include <cudautils/cudautils-vecmath.h>
#include <cudautils/cudautils-film.h>

namespace dmt {
    struct DeviceCamera
    {
        float  focalLength = 20.f;
        float  sensorSize  = 36.f;
        float3 dir{0.f, 1.f, 0.f};
        //float camDirX     = 0.f;
        //float camDirY     = 1;
        //float camDirZ     = 0.f;
        int spp = 4;
        //float camPosX     = 0.f;
        //float camPosY     = 0.f;
        //float camPosZ     = 0.f;
        float3 pos{0.f, 0.f, 0.f};
        int    width  = 128;
        int    height = 128;
    };

    DMT_FORCEINLINE DMT_GPU void generate_camera_ray(
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
        float&       rdz)
    {
        // Basic pinhole: raster->NDC jitter
        float    fx  = (px + u1) / float(cam.width);
        float    fy  = (py + u2) / float(cam.height);
        Vector3f ndc = {2.f * fx - 1.f, 1.f - 2.f * fy, 1.f};

        // Assume renderFromCam encodes a canonical camera at origin looking -Z; adapt to your transforms.
        // For now: simple camera at origin:
        rox = roy = roz = 0.f;

        rdx = ndc.x;
        rdy = ndc.y;
        rdz = -1.f;
    }


} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_CAMERA_IMPL)
    #include "cudautils-camera.cu"
#endif