#include "cudautils-camera.h"

namespace dmt {
    // CameraTransform Method Definitions
    CameraTransform::CameraTransform(AnimatedTransform const& worldFromCamera)
    {
        // Compute _worldFromRender_ for camera-world space rendering
        float   tMid    = (worldFromCamera.startTime + worldFromCamera.endTime) / 2;
        Point3f pCamera = worldFromCamera(Point3f(Tuple3(0.f, 0.f, 0.f)), tMid);
        worldFromRender = Translate(Vector3f(pCamera));


        //LOG_VERBOSE("World-space position: %s", worldFromRender(Point3f(0, 0, 0)));
        // Compute _renderFromCamera_ transformation
        Transform renderFromWorld = Inverse(worldFromRender);
        Transform rfc[2] = {renderFromWorld * worldFromCamera.startTransform, renderFromWorld * worldFromCamera.endTransform};
        renderFromCamera = AnimatedTransform(rfc[0], worldFromCamera.startTime, rfc[1], worldFromCamera.endTime);
    }

    std::string CameraTransform::ToString() const
    {
        /*return "[ CameraTransform renderFromCamera: %s worldFromRender: %s ]", renderFromCamera, worldFromRender);*/
        return "";
    }

    DMT_CPU_GPU dstd::optional<CameraRayDifferential> CameraBase::GenerateRayDifferential(
        CameraBase          camera,
        CameraSample        sample,
        SampledWavelengths& lambda)
    {
        // Generate regular camera ray _cr_ for ray differential
        dstd::optional<CameraRay> cr = camera.GenerateRay ? camera.GenerateRay(sample, lambda) : dstd::optional<CameraRay>();
        if (!cr)
            return {};
        RayDifferential rd(cr->ray);

        // Find camera ray after shifting one pixel in the $x$ direction
        dstd::optional<CameraRay> rx;
        for (float eps : {.05f, -.05f})
        {
            CameraSample sshift = sample;
            sshift.pFilm.x += eps;
            // Try to generate ray with _sshift_ and compute $x$ differential
            if (rx = camera.GenerateRay ? camera.GenerateRay(sshift, lambda) : dstd::optional<CameraRay>(); rx)
            {
                rd.rxOrigin    = rd.o + (rx->ray.o - rd.o) / eps;
                rd.rxDirection = rd.d + (rx->ray.d - rd.d) / eps;
                break;
            }
        }

        // Find camera ray after shifting one pixel in the $y$ direction
        dstd::optional<CameraRay> ry;
        for (float eps : {.05f, -.05f})
        {
            CameraSample sshift = sample;
            sshift.pFilm.y += eps;
            if (ry = camera.GenerateRay ? camera.GenerateRay(sshift, lambda) : dstd::optional<CameraRay>(); ry)
            {
                rd.ryOrigin    = rd.o + (ry->ray.o - rd.o) / eps;
                rd.ryDirection = rd.d + (ry->ray.d - rd.d) / eps;
                break;
            }
        }

        // Return approximate ray differential and weight
        rd.hasDifferentials = rx && ry;
        return CameraRayDifferential{rd, cr->weight};
    }
} // namespace dmt