#pragma once

#include "dmtmacros.h"

#include <platform/platform-utils.h>
#include "cudautils/cudautils-float.h"

// silence warnings __host__ __device__ on a defaulted copy control
#if defined(__NVCC__)
#pragma nv_diag_suppress 20012
#endif
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/vec3.hpp>   // Vec3f
#include <glm/vec4.hpp>   // Vec4f
#include <glm/ext/quaternion_float.hpp>
#if defined(__NVCC__)
#pragma nv_diag_default 20012
#endif

#include <array>

#include <cassert>
#include <cstdint>

namespace dmt {
    /**
     * Tag which signals that the containing class should have a correspondance inside the `soa` namespace
     */
    struct SOA {};

    // Enums ----------------------------------------------------------------------------------------------------------
    enum class ERenderCoordSys : uint8_t
    {
        eCameraWorld = 0,
        eCamera,
        eWorld,
        eCount
    };


    // math utilities -------------------------------------------------------------------------------------------------
    enum class EVecType32 : uint32_t
    {
        eVector = 0,
        ePoint  = 0x3f80'0000, // 1.f
    };
    namespace fl {
        DMT_CPU_GPU float inline bitsToFloat(EVecType32 e) {
            return bitsToFloat(static_cast<std::underlying_type_t<EVecType32>>(e));
        }
    }

    // TODO SOA
    struct Intervalf
    {
        friend SOA;
    public:
        Intervalf() = default;
        DMT_CPU_GPU explicit Intervalf(float v);
        DMT_CPU_GPU                  Intervalf(float low, float high);
        DMT_CPU_GPU static Intervalf fromValueAndError(float v, float err);

    public:
        DMT_CPU_GPU float midpoint() const;
        DMT_CPU_GPU float width() const;

    public:
        float low  = 0.f;
        float high = 0.f;
    };
    DMT_CPU_GPU Intervalf operator+(Intervalf a, Intervalf b);
    DMT_CPU_GPU Intervalf operator-(Intervalf a, Intervalf b);
    DMT_CPU_GPU Intervalf operator/(Intervalf a, Intervalf b);
    DMT_CPU_GPU Intervalf operator*(Intervalf a, Intervalf b);

    // vectors

    using Vec3f  = glm::vec3;
    using Pt3f   = Vec3f;
    using Norm3f = Vec3f; // how to make sure that it always have unit length? Store compressed?
    using Vec4f  = glm::vec4;
    using Mat4f  = glm::mat4;
    using Quat   = glm::quat;

    class Transform
    {
    public:
        Mat4f m;    // Transformation matrix
        Mat4f mInv; // Inverse transformation matrix

        // Default constructor
        DMT_CPU_GPU Transform();

        // Constructor with an initial matrix
        DMT_CPU_GPU explicit Transform(Mat4f const& matrix);

        // Apply translation
        DMT_CPU_GPU void translate_(Vec3f const& translation);

        // Apply scaling
        DMT_CPU_GPU void scale_(Vec3f const& scaling);

        // Apply rotation (angle in degrees)
        DMT_CPU_GPU void rotate_(float angle, Vec3f const& axis);

        // Combine with another transform
        DMT_CPU_GPU Transform combine(Transform const& other) const;

        // Combine with another transform
        DMT_CPU_GPU void combine_(Transform const& other);

        DMT_CPU_GPU void lookAt_(Vec3f pos, Vec3f look, Vec3f up);

        DMT_CPU_GPU void concatTrasform_(std::array<float, 16> const& transform);

        // Reset to identity matrix
        DMT_CPU_GPU void reset_();

        // Swap m and mInv
        DMT_CPU_GPU void inverse_();

        // Apply the transform to a point
        DMT_CPU_GPU Vec3f applyToPoint(Vec3f const& point) const;

        // Apply the inverse transform to a point
        DMT_CPU_GPU Vec3f applyInverseToPoint(Vec3f const& point) const;

        DMT_CPU_GPU void decompose(Vec3f& outT, Quat& outR, Mat4f& outS) const;

        // Equality comparison
        DMT_CPU_GPU bool operator==(Transform const& other) const;

        // Inequality comparison
        DMT_CPU_GPU bool operator!=(Transform const& other) const;

        DMT_CPU_GPU bool     hasScale(float tolerance = 1e-3f) const;
        DMT_CPU_GPU Vec3f    applyInverse(Vec3f vpn, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU Vec3f    operator()(Vec3f vpn, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU struct Bounds3f operator()(struct Bounds3f const& b) const;
    };

    struct Bounds3f
    {
        enum EBoundsCorner : int32_t
        {
            eBottom  = 0,
            eLeft    = eBottom,
            eBack    = eLeft,
            eTop     = 0b100,
            eRight   = 0b001,
            eForward = 0b010,
            // 8 corners
            eRightTopForward    = eRight | eTop | eForward,
            eRightTopBack       = eRight | eTop | eBack,
            eRightBottomForward = eRight | eBottom | eForward,
            eRightBottomBack    = eRight | eBottom | eBack,
            eLeftTopForward     = eLeft | eTop | eForward,
            eLeftTopBack        = eLeft | eTop | eBack,
            eLeftBottomForward  = eLeft | eBottom | eForward,
            eLeftBottomBack     = eLeft | eBottom | eBack,
        };

        DMT_CPU_GPU Vec3f&       operator[](int32_t i);
        DMT_CPU_GPU Vec3f const& operator[](int32_t i) const;
        DMT_CPU_GPU Pt3f         corner(EBoundsCorner corner) const;
        DMT_CPU_GPU Vec3f        diagonal() const;
        DMT_CPU_GPU float        surfaceAraa() const;
        DMT_CPU_GPU float        volume() const;
        DMT_CPU_GPU int32_t      maxDimention() const;
        DMT_CPU_GPU Pt3f         lerp(Pt3f t) const;
        DMT_CPU_GPU Vec3f        offset(Pt3f p) const;
        DMT_CPU_GPU void         boundingSphere(Pt3f& outCenter, float& outRadius) const;
        DMT_CPU_GPU bool         isEmpty() const;
        DMT_CPU_GPU bool         isDegenerate() const;
        DMT_CPU_GPU bool         operator==(Bounds3f const& that) const;
        DMT_CPU_GPU bool         intersectP(Pt3f                o,
                                            Vec3f               d,
                                            float               tMax = std::numeric_limits<float>::infinity(),
                                            float* DMT_RESTRICT hit0 = nullptr,
                                            float* DMT_RESTRICT hit1 = nullptr) const;
        DMT_CPU_GPU bool intersectP(Pt3f o, Vec3f d, float tMax, Vec3f invDir, std::array<int32_t, 3> dirIsNeg) const;

        Vec3f pMin;
        Vec3f pMax;
    };

    DMT_CPU_GPU bool     inside(Pt3f p, Bounds3f const& b);
    DMT_CPU_GPU bool     almostEqual(Mat4f const& a, Mat4f const& b);
    DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& a, Bounds3f const& b);
    DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& b, Pt3f p);

    class AnimatedTransform
    {
    public:
        AnimatedTransform() = default;
        DMT_CPU_GPU AnimatedTransform(Transform const& startTransform, float startTime, Transform const& endTransform, float endTime);

        DMT_CPU_GPU bool  isAnimated() const;
        DMT_CPU_GPU Vec3f applyInverse(Vec3f vpn, float time, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU Vec3f operator()(Vec3f vpn, float time, EVecType32 type, bool normalize = false) const;
        // TODO: Interaction methods

        DMT_CPU_GPU bool      hasScale() const;
        DMT_CPU_GPU bool      hasRotation() const;
        DMT_CPU_GPU Transform interpolate(float time) const;
        // TODO: Ray and RayDifferential methods
        DMT_CPU_GPU Bounds3f motionBounds(Bounds3f const& b) const;
        DMT_CPU_GPU Bounds3f boundsPointMotion(Pt3f p) const;

    private:
        DMT_CPU_GPU static void findZeros(
            float            c1,
            float            c2,
            float            c3,
            float            c4,
            float            c5,
            float            theta,
            Intervalf        tInterval,
            ArrayView<float> zeros,
            int32_t&         outNumZeros,
            int32_t          depth = 8);

    public:
        Transform startTransform;
        Transform endTransform;
        float     startTime = 0;
        float     endTime   = 1;

    private:
        struct DerivativeTerm
        {
            DMT_CPU_GPU       DerivativeTerm();
            DMT_CPU_GPU       DerivativeTerm(float c, float x, float y, float z);
            DMT_CPU_GPU float eval(Pt3f p) const;

            float kc, kx, ky, kz;
        };

        enum EState : int32_t
        {
            eNone        = 0,
            eAnimated    = 1,
            eHasRotation = 2,
            eAll         = eAnimated | eHasRotation,
        };

        // rigid transformations
        DerivativeTerm m_c1[3], m_c2[3], m_c3[3], m_c4[3], m_c5[3];
        Mat4f          m_s[2];
        Quat           m_r[2];
        Vec3f          m_t[2];
        EState         m_state;
    };

    struct CameraTransform
    {
        // requires initialized context
        DMT_CPU_GPU       CameraTransform(AnimatedTransform const& worldFromCamera, ERenderCoordSys renderCoordSys);
        AnimatedTransform renderFromCamera;
        Transform         worldFromRender;
    };

} // namespace dmt

namespace dmt::soa {
    using namespace dmt;
}