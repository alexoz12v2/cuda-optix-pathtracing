#include "cudautils/cudautils-camera.cuh"

namespace dmt {
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