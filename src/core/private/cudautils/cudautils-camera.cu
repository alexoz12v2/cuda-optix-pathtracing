#include "cudautils/cudautils-camera.cuh"

namespace dmt {
    DMT_GPU void generate_camera_ray(DeviceCamera cam, int px, int py, float u1, float u2, float3& ro, float3& rd)
    {
        // Basic pinhole: raster->NDC jitter
        float    fx  = (px + u1) / float(cam.width);
        float    fy  = (py + u2) / float(cam.height);
        Vector3f ndc = {2.f * fx - 1.f, 1.f - 2.f * fy, 1.f};

        // Assume renderFromCam encodes a canonical camera at origin looking -Z; adapt to your transforms.
        // For now: simple camera at origin:
        ro.x = ro.y = ro.z = 0.f;

        rd.x = ndc.x;
        rd.y = ndc.y;
        rd.z = -1.f;
    }
} // namespace dmt