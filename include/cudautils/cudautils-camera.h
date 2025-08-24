#pragma once

#include "cudautils/cudautils-macro.h"
#include <cudautils/cudautils-spectrum.h>
#include <cudautils/cudautils-transform.h>
#include <cudautils/cudautils-vecmath.h>
#include <cudautils/cudautils-film.h>

namespace dmt {
    struct DMT_CORE_API CameraSample
    {
        Point2f pFilm;
        Point2f pLens;
        float   time         = 0;
        float   filterWeight = 1;
    };

    struct DMT_CORE_API CameraRay
    {
        Ray             ray;
        SampledSpectrum weight = SampledSpectrum(1);
    };

    struct DMT_CORE_API CameraRayDifferential
    {
        RayDifferential ray;
        SampledSpectrum weight = SampledSpectrum(1);
    };

    // CameraTransform Definition
    class DMT_CORE_API CameraTransform
    {
    public:
        // CameraTransform Public Methods
        CameraTransform() = default;
        explicit CameraTransform(AnimatedTransform const& worldFromCamera);


        DMT_CPU_GPU Point3f RenderFromCamera(Point3f p, float time) const { return renderFromCamera(p, time); }

        DMT_CPU_GPU Point3f CameraFromRender(Point3f p, float time) const
        {
            return renderFromCamera.applyInverse(p, time);
        }

        DMT_CPU_GPU Point3f RenderFromWorld(Point3f p) const { return worldFromRender.applyInverse(p); }


        DMT_CPU_GPU Transform RenderFromWorld() const { return Inverse(worldFromRender); }

        DMT_CPU_GPU Transform CameraFromRender(float_t time) const
        {
            return Inverse(renderFromCamera.interpolate(time));
        }

        DMT_CPU_GPU Transform CameraFromWorld(float time) const
        {
            return Inverse(worldFromRender * renderFromCamera.interpolate(time));
        }


        DMT_CPU_GPU bool CameraFromRenderHasScale() const { return renderFromCamera.hasScale(); }


        DMT_CPU_GPU Vector3f RenderFromCamera(Vector3f v, float time) const { return renderFromCamera(v, time); }


        DMT_CPU_GPU Normal3f RenderFromCamera(Normal3f n, float time) const { return renderFromCamera(n, time); }


        DMT_CPU_GPU Ray RenderFromCamera(Ray const& r) const { return renderFromCamera(r); }


        DMT_CPU_GPU RayDifferential RenderFromCamera(RayDifferential const& r) const { return renderFromCamera(r); }


        DMT_CPU_GPU Vector3f CameraFromRender(Vector3f v, float time) const
        {
            return renderFromCamera.applyInverse(v, time);
        }


        DMT_CPU_GPU Normal3f CameraFromRender(Normal3f v, float time) const
        {
            return renderFromCamera.applyInverse(v, time);
        }


        DMT_CPU_GPU AnimatedTransform const& RenderFromCamera() const { return renderFromCamera; }


        DMT_CPU_GPU Transform const& WorldFromRender() const { return worldFromRender; }

        std::string ToString() const;

    private:
        // CameraTransform Private Members
        AnimatedTransform renderFromCamera;
        Transform         worldFromRender;
    };

    struct DMT_CORE_API CameraBaseParameters
    {
        CameraTransform cameraTransform;
        float           shutterOpen = 0, shutterClose = 1;
        Film            film;

        CameraBaseParameters() = default;
        //see better
        CameraBaseParameters(CameraTransform const& cameraTransorm, Film film);
    };

    class DMT_CORE_API CameraBase
    {
    public:
        Film                   GetFilm() const { return film; }
        CameraTransform const& GetCametaTransform() const { return cameraTransform; }

        float SampleTime(float u) const { return lerp(u, shutterOpen, shutterClose); }
        /*Capiamo*/
        //void        InitMetadata(ImageMetadata* metadata) const;
        std::string ToString() const;
        void Approxiamate_dp_dxy(Point3f p, Normal3f n, float time, int samplesPerPixel, Vector3f* dpdx, Vector3f* dpdy) const
        {
            Point3f   pCamera         = CameraFromRender(p, time);
            Transform DownZFromCamera = Transform::rotateFromTo(normalize(Vector3f(pCamera)), Vector3f{{0, 0, 1}});
            Point3f   pDownZ          = DownZFromCamera(pCamera);
            Normal3f  nDownZ          = DownZFromCamera(CameraFromRender(n, time));
            float     d               = nDownZ.z * pDownZ.z;
            Ray   xRay{Point3f{{0, 0, 0}} + minPosDifferentialX, normalize(Vector3f{{0, 0, 1}} + minDirDifferentialX)};
            float tx = -(dot(nDownZ, Vector3f(xRay.o)) - d) / dot(nDownZ, xRay.d);
            Ray   yRay(Point3f{{0, 0, 0}} + minPosDifferentialY, normalize(Vector3f{{0, 0, 1}} + minDirDifferentialY));
            float ty   = -(dot(nDownZ, Vector3f(yRay.o)) - d) / dot(nDownZ, yRay.d);
            Point3f px = xRay(tx), py = yRay(ty);
            /*//PixelJitter is an option but now I don't know if handle so for now is enable
            float     sppScale = GetOptions().disablePixelJitter
                                     ? 1
                                     : std::max<float>(.125, 1 / std::sqrt((float)samplesPerPixel));
            */
            float sppScale = std::max<float>(.125, 1 / std::sqrt((float)samplesPerPixel));

            *dpdx = sppScale * RenderFromCamera(DownZFromCamera.applyInverse(px - pDownZ), time);
            *dpdy = sppScale * RenderFromCamera(DownZFromCamera.applyInverse(py - pDownZ), time);
        }

    protected:
        CameraTransform cameraTransform;
        float           shutterOpen, shutterClose;
        Film            film;
        Vector3f        minPosDifferentialX, minPosDifferentialY;
        Vector3f        minDirDifferentialX, minDirDifferentialY;

        CameraBase() = default;
        CameraBase(CameraBaseParameters p);

        // In derived
        DMT_CPU_GPU dstd::optional<CameraRay> (*GenerateRay)(CameraSample sample, SampledWavelengths& lambda) = nullptr;

        DMT_CPU_GPU static dstd::optional<CameraRayDifferential> GenerateRayDifferential(CameraBase          camera,
                                                                                         CameraSample        sample,
                                                                                         SampledWavelengths& lambda);

        DMT_CPU_GPU Ray RenderFromCamera(Ray const& r) const { return cameraTransform.RenderFromCamera(r); }

        DMT_CPU_GPU RayDifferential RenderFromCamera(RayDifferential const& r) const
        {
            return cameraTransform.RenderFromCamera(r);
        }


        DMT_CPU_GPU Vector3f RenderFromCamera(Vector3f v, float time) const
        {
            return cameraTransform.RenderFromCamera(v, time);
        }


        DMT_CPU_GPU Normal3f RenderFromCamera(Normal3f v, float time) const
        {
            return cameraTransform.RenderFromCamera(v, time);
        }


        DMT_CPU_GPU Point3f RenderFromCamera(Point3f p, float time) const
        {
            return cameraTransform.RenderFromCamera(p, time);
        }


        DMT_CPU_GPU Vector3f CameraFromRender(Vector3f v, float time) const
        {
            return cameraTransform.CameraFromRender(v, time);
        }


        DMT_CPU_GPU Normal3f CameraFromRender(Normal3f v, float time) const
        {
            return cameraTransform.CameraFromRender(v, time);
        }


        DMT_CPU_GPU Point3f CameraFromRender(Point3f p, float time) const
        {
            return cameraTransform.CameraFromRender(p, time);
        }

        //TODO: integrate it
        //void FindMinimumDifferentials(Camera camera);
    };

    //ProjectiveCamera Definition
    class DMT_CORE_API ProjectiveCamera : public CameraBase
    {
    public:
        ProjectiveCamera() = default;
        /*Capiamo*/
        //void        InitMetadata(ImageMetadata* metadata) const;

        std::string BaseToString() const;
        //ProjectiveCamera(CameraBaseParameters baseParameters, Transform const& screenFromCamera);
    };


} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_CAMERA_IMPL)
    #include "cudautils-camera.cu"
#endif