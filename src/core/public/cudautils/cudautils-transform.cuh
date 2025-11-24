#pragma once

#include "core/cudautils/cudautils-macro.cuh"
#include "core/cudautils/cudautils-enums.cuh"
#include "core/cudautils/cudautils-vecmath.cuh"

namespace dmt {
    // Transform, AnimatedTransform, CameraTransform ------------------------------------------------------------------
    class DMT_CORE_API alignas(16) Transform
    {
    public:
        Matrix4f m;    // Transformation matrix
        Matrix4f mInv; // Inverse transformation matrix

        // Default constructor
        DMT_CPU_GPU Transform() = default;

        // Constructor with an initial matrix
        DMT_CPU_GPU explicit Transform(Matrix4f const& matrix);

        // Apply translation
        static DMT_CPU_GPU Transform translate(Vector3f const& translation);

        // Apply scaling
        static DMT_CPU_GPU Transform scale(float scaling) { return scale({scaling, scaling, scaling}); }
        static DMT_CPU_GPU Transform scale(Vector3f const& scaling);

        // Apply rotation (angle in degrees)
        static DMT_CPU_GPU Transform rotate(float angle, Vector3f const& axis);

        /// Rodrigues' Rotation formula
        static DMT_CPU_GPU Transform rotateFromTo(Vector3f from, Vector3f to);

        // Apply translation
        DMT_CPU_GPU void translate_(Vector3f const& translation);

        // Apply scaling
        DMT_CPU_GPU void scale_(Vector3f const& scaling);

        // Apply rotation (angle in degrees)
        DMT_CPU_GPU void rotate_(float angle, Vector3f const& axis);

        // Combine with another transform
        DMT_CPU_GPU Transform combine(Transform const& other) const;

        // Combine with another transform
        DMT_CPU_GPU void combine_(Transform const& other);

        DMT_CPU_GPU void lookAt_(Vector3f pos, Vector3f look, Vector3f up);

        DMT_CPU_GPU void concatTrasform_(std::array<float, 16> const& transform);

        // Reset to identity matrix
        DMT_CPU_GPU void reset_();

        // Swap m and mInv
        DMT_CPU_GPU void inverse_();

        DMT_CPU_GPU void decompose(Vector3f& outT, Quaternion& outR, Matrix4f& outS) const;

        // Equality comparison
        DMT_CPU_GPU bool operator==(Transform const& other) const;

        // Inequality comparison
        DMT_CPU_GPU bool operator!=(Transform const& other) const;

        DMT_CPU_GPU bool            hasScale(float tolerance = 1e-3f) const;
        DMT_CPU_GPU Vector3f        applyInverse(Vector3f v) const;
        DMT_CPU_GPU Point3f         applyInverse(Point3f v) const;
        DMT_CPU_GPU Normal3f        applyInverse(Normal3f v) const;
        DMT_CPU_GPU Point3fi        applyInverse(Point3fi v) const;
        DMT_CPU_GPU Ray             applyInverse(Ray const& ray, float* optInOut_tMax) const;
        DMT_CPU_GPU RayDifferential applyInverse(RayDifferential const& ray, float* optInOut_tMax) const;
        DMT_CPU_GPU Vector3f        operator()(Vector3f v) const;
        DMT_CPU_GPU Point3f         operator()(Point3f v) const;
        DMT_CPU_GPU Point3fi        operator()(Point3fi v) const;
        DMT_CPU_GPU Normal3f        operator()(Normal3f v) const;
        DMT_CPU_GPU Bounds3f        operator()(Bounds3f const& b) const;
        DMT_CPU_GPU Ray             operator()(Ray const& ray, float* optInOut_tMax) const;
        DMT_CPU_GPU RayDifferential operator()(RayDifferential const& ray, float* optInOut_tMax) const;

        DMT_CPU_GPU bool swapsHandedness() const;
    };

    // Transform Function Declarations
    DMT_CORE_API DMT_CPU_GPU Transform Translate(Vector3f delta);

    // Transform Inline Functions
    DMT_CPU_GPU inline Transform Inverse(Transform const& t) { return Transform(t.m); }

#if defined(__CUDA_ARCH__)
    DMT_FORCEINLINE
#else
    DMT_FORCEINLINE inline
#endif
    DMT_CPU_GPU Transform operator*(Transform const& a, Transform const& b) { return a.combine(b); }

    class DMT_CORE_API AnimatedTransform
    {
    public:
        AnimatedTransform() = default;
        DMT_CPU_GPU AnimatedTransform(Transform const& startTransform, float startTime, Transform const& endTransform, float endTime);

        DMT_CPU_GPU bool     isAnimated() const;
        DMT_CPU_GPU Vector3f applyInverse(Vector3f vpn, float time) const;
        DMT_CPU_GPU Point3f  applyInverse(Point3f vpn, float time) const;
        DMT_CPU_GPU Normal3f applyInverse(Normal3f vpn, float time) const;
        DMT_CPU_GPU Vector3f operator()(Vector3f vpn, float time) const;
        DMT_CPU_GPU Point3f  operator()(Point3f vpn, float time) const;
        DMT_CPU_GPU Normal3f operator()(Normal3f vpn, float time) const;


        // ray encapsulates time
        DMT_CPU_GPU Ray             operator()(Ray const& ray, float* optInOut_tMax = nullptr) const;
        DMT_CPU_GPU RayDifferential operator()(RayDifferential const& ray, float* optInOut_tMax = nullptr) const;
        DMT_CPU_GPU Ray             applyInverse(Ray const& ray, float* optInOut_tMax) const;
        DMT_CPU_GPU RayDifferential applyInverse(RayDifferential const& ray, float* optInOut_tMax) const;
        // TODO: Interaction methods

        DMT_CPU_GPU bool      hasScale() const;
        DMT_CPU_GPU bool      hasRotation() const;
        DMT_CPU_GPU Transform interpolate(float time) const;
        // TODO: Ray and RayDifferential methods
        DMT_CPU_GPU Bounds3f motionBounds(Bounds3f const& b) const;
        DMT_CPU_GPU Bounds3f boundsPointMotion(Point3f p) const;

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
        struct DMT_CORE_API DerivativeTerm
        {
            DMT_CPU_GPU       DerivativeTerm();
            DMT_CPU_GPU       DerivativeTerm(float c, float x, float y, float z);
            DMT_CPU_GPU float eval(Point3f p) const;

            float kc, kx, ky, kz;
        };

        enum DMT_CORE_API EState : int32_t
        {
            eNone        = 0,
            eAnimated    = 1,
            eHasRotation = 2,
            eAll         = eAnimated | eHasRotation,
        };

        // rigid transformations
        DerivativeTerm m_c1[3], m_c2[3], m_c3[3], m_c4[3], m_c5[3];
        Matrix4f       m_s[2]{Matrix4f::identity()};
        Quaternion     m_r[2]{Quaternion::quatIdentity()};
        Vector3f       m_t[2]{Vector3f::zero()};
        EState         m_state = eNone;
    };

} // namespace dmt
