#pragma once

#include "cudautils/cudautils-macro.h"

#include "cudautils/cudautils-enums.h"
#include "cudautils/cudautils-vecmath.h"

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

#if defined(DMT_CUDAUTILS_IMPLEMENTATION)
    // Private Static Stuff -------------------------------------------------------------------------------------------
    static __host__ __device__ void adjustRangeToErrorBounds(Vector3f dir, Point3fi& o, float* optInOut_tMax)
    {
        // offset ray origin to the edge of the error bounds and adjust the `tMax`
        if (float len2 = dotSelf(dir); len2 > 0.f)
        {
            float dt = dot(abs(dir), o.error()) / len2;
            o        = o + Point3fi(dir * dt); // TODO: support for more operations with intervals
            if (optInOut_tMax)
                *optInOut_tMax -= dt;
        }
    }

    __host__ __device__ Transform::Transform(Matrix4f const& matrix) : m(matrix), mInv(inverse(matrix)) {}

    __host__ __device__ void Transform::translate_(Vector3f const& translation)
    {
        alignas(16) glm::mat4 gm    = glm::translate(*toGLMmat(&m), *toGLM(&translation));
        alignas(16) glm::mat4 gminv = glm::translate(*toGLMmat(&mInv), -*toGLM(&translation));

        m    = *fromGLMmat(&gm);
        mInv = *fromGLMmat(&gminv);
    }

    // Apply scaling
    __host__ __device__ void Transform::scale_(Vector3f const& scaling)
    {
        Vector3f const        vec   = bcast<Vector3f>(1.0f) / scaling;
        alignas(16) glm::mat4 gm    = glm::scale(*toGLMmat(&m), *toGLM(&scaling));
        alignas(16) glm::mat4 gminv = glm::scale(*toGLMmat(&mInv), *toGLM(&vec));

        m    = *fromGLMmat(&gm);
        mInv = *fromGLMmat(&gminv);
    }

    // Apply rotation (angle in degrees)
    __host__ __device__ void Transform::rotate_(float angle, Vector3f const& axis)
    {
        alignas(16) glm::mat4 gm    = glm::rotate(*toGLMmat(&m), glm::radians(angle), *toGLM(&axis));
        alignas(16) glm::mat4 gminv = glm::rotate(*toGLMmat(&mInv), -glm::radians(angle), *toGLM(&axis));

        m    = *fromGLMmat(&gm);
        mInv = *fromGLMmat(&gminv);
    }

    // Combine with another transform
    __host__ __device__ Transform Transform::combine(Transform const& other) const
    {
        Transform result;
        result.m    = m * other.m;
        result.mInv = other.mInv * mInv; // (AB)^-! = B^-1 A^-1
        return result;
    }

    __host__ __device__ Transform Transform::translate(Vector3f const& translation)
    {
        Transform t;
        t.translate_(translation);
        return t;
    }

    __host__ __device__ Transform Transform::scale(Vector3f const& scaling)
    {
        Transform t;
        t.scale_(scaling);
        return t;
    }

    __host__ __device__ Transform Transform::rotate(float angle, Vector3f const& axis)
    {
        Transform t;
        t.rotate_(angle, axis);
        return t;
    }

    __host__ __device__ Transform Transform::rotateFromTo(Vector3f from, Vector3f to)
    {
        assert(fl::abs(normL2(from) - 1) < 1e-5f && fl::abs(normL2(to) - 1) < 1e-5f);
        static constexpr float eps = 1e-6f;

        float const cosTheta = dot(from, to);
        if (cosTheta > 1.f - eps) // case 1: vectorsa already aligned
            return Transform{};
        else if (cosTheta < -1.f + eps) // case 2: vectors are the opposite
        {
            Vector3f orthogonal = Vector3f(1.0f, 0.0f, 0.0f);
            if (absDot(from, orthogonal) > 0.99f)
                orthogonal = Vector3f(0.0f, 1.0f, 0.0f);
            Vector3f axis = normalize(cross(from, orthogonal));
            float    x = axis.x, y = axis.y, z = axis.z;
            float    xx = x * x, yy = y * y, zz = z * z;
            float    xy = x * y, yz = y * z, zx = z * x;
            float    c = -1.0f;
            float    t = 1 - c;

            // clang-format off
            Matrix4f R{
                t * xx + c, t * xy - z, t * zx + y, 0,
                t * xy + z, t * yy + c, t * yz - x, 0,
                t * zx - y, t * yz + x, t * zz + c, 0,
                0, 0, 0, 1
            };
            // clang-format on
            return Transform(R);
        }
        else // general case
        {
            Vector3f const v         = cross(from, to);
            float const    s         = normL2(v);
            float const    oneMinusC = 1.f - cosTheta;

            float x = v.x, y = v.y, z = v.z;
            float vx[9] = {0, -z, y, z, 0, -x, -y, x, 0};

            // Compute [v]_x^2
            // clang-format off
            float vx2[9] = {
                -y*y - z*z, x*y,       x*z, 
                x*y,       -x*x - z*z, y*z, 
                x*z,        y*z,      -x*x - y*y };
            // clang-format on

            // Rodrigues' formula: R = I + [v]_x + [v]_x^2 * ((1 - c) / s^2)
            float    k = oneMinusC / (s * s);
            Matrix4f R = Matrix4f::identity();
            R.m[0] += vx[0] + vx2[0] * k;
            R.m[4] += vx[1] + vx2[1] * k;
            R.m[8] += vx[2] + vx2[2] * k;
            R.m[1] += vx[3] + vx2[3] * k;
            R.m[5] += vx[4] + vx2[4] * k;
            R.m[9] += vx[5] + vx2[5] * k;
            R.m[2] += vx[6] + vx2[6] * k;
            R.m[6] += vx[7] + vx2[7] * k;
            R.m[10] += vx[8] + vx2[8] * k;

            return Transform(R);
        }
    }

    // Combine with another transform
    __host__ __device__ void Transform::combine_(Transform const& other)
    {
        m    = m * other.m;
        mInv = other.mInv * mInv;
    }

    __host__ __device__ void Transform::lookAt_(Vector3f pos, Vector3f look, Vector3f up)
    {
        Matrix4f worldFromCamera;
        // Initialize fourth column of viewing matrix
        worldFromCamera[{0, 3}] = pos.x;
        worldFromCamera[{1, 3}] = pos.y;
        worldFromCamera[{2, 3}] = pos.z;
        worldFromCamera[{3, 3}] = 1;

        // Initialize first three columns of viewing matrix
        glm::vec3 nup     = glm::normalize(*toGLM(&up));
        glm::vec3 dir     = glm::normalize(*toGLM(&look) - *toGLM(&pos));
        glm::vec3 nupXdir = glm::cross(nup, dir);
        assert(glm::length2(nupXdir) > std::numeric_limits<float>::epsilon());

        glm::vec3 right         = glm::normalize(nupXdir);
        glm::vec3 newUp         = glm::cross(dir, right);
        worldFromCamera[{0, 0}] = right.x;
        worldFromCamera[{1, 0}] = right.y;
        worldFromCamera[{2, 0}] = right.z;
        worldFromCamera[{3, 0}] = 0.f;
        worldFromCamera[{0, 1}] = newUp.x;
        worldFromCamera[{1, 1}] = newUp.y;
        worldFromCamera[{2, 1}] = newUp.z;
        worldFromCamera[{3, 1}] = 0.f;
        worldFromCamera[{0, 2}] = dir.x;
        worldFromCamera[{1, 2}] = dir.y;
        worldFromCamera[{2, 2}] = dir.z;
        worldFromCamera[{3, 2}] = 0.f;

        m    = m * worldFromCamera;
        mInv = inverse(worldFromCamera) * mInv;
    }

    // We expect a row major array, hence we transpose it
    __host__ __device__ void Transform::concatTrasform_(std::array<float, 16> const& t)
    {
        Matrix4f mt{
            {t[0], t[1], t[2], t[3], /**/ t[4], t[5], t[6], t[7], /**/ t[8], t[9], t[10], t[11], /**/ t[12], t[13], t[14], t[15]}};
        Transform concatT{transpose(m)};
        m    = m * concatT.m;
        mInv = concatT.mInv * mInv;
    }

    __host__ __device__ void Transform::reset_()
    {
        m    = Matrix4f::identity();
        mInv = Matrix4f::identity();
    }

    __host__ __device__ void Transform::inverse_()
    {
        Matrix4f tmp = m;
        m            = mInv;
        mInv         = tmp;
    }


    __host__ __device__ void Transform::decompose(Vector3f& outT, Quaternion& outR, Matrix4f& outS) const
    {
        // discarded components
        glm::vec3 scale;
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(*toGLMmat(&m), scale, *toGLM(&outR), *toGLM(&outT), skew, perspective);
        // decompose actually returs the conjugate quaternion
        glm::quat conj = glm::conjugate(*toGLM(&outR));
        outR           = *fromGLMquat(&conj);
        // inglobe all the rest into a matrixc
        // Start with an identity matrix
        outS = Matrix4f::identity();

        // Apply scaling
        outS[{0, 0}] = scale.x;
        outS[{1, 1}] = scale.y;
        outS[{2, 2}] = scale.z;

        // Apply skew (off-diagonal elements)
        outS[{1, 0}] = skew.x; // Skew Y by X
        outS[{2, 0}] = skew.y; // Skew Z by X
        outS[{2, 1}] = skew.z; // Skew Z by Y

        // Apply perspective (set the last row)
        outS[3] = *fromGLM(&perspective);
    }


    // Equality comparison
    __host__ __device__ bool Transform::operator==(Transform const& other) const
    {
        return near(m, other.m) && near(mInv, other.mInv);
    }

    // Inequality comparison
    __host__ __device__ bool Transform::operator!=(Transform const& other) const { return !(*this == other); }

    __host__ __device__ bool Transform::hasScale(float tolerance) const
    {
        // compute the length of the three reference unit vectors after being transformed. if any of these has been
        // scaled, then the transformation has a scaling component
        float const    x = dotSelf(operator()(Vector3f{{1.f, 0.f, 0.f}}));
        float const    y = dotSelf(operator()(Vector3f{{0.f, 1.f, 0.f}}));
        float const    z = dotSelf(operator()(Vector3f{{0.f, 0.f, 1.f}}));
        Vector3f const scales{{x, y, z}};

        bool res = !near(scales, Vector3f{{1.f, 1.f, 1.f}});
        return res;
    }

    __host__ __device__ Vector3f Transform::applyInverse(Vector3f v) const { return mul(mInv, v); }
    __host__ __device__ Point3f  Transform::applyInverse(Point3f v) const { return mul(mInv, v); }
    // TODO method mulTranspose instead of storing the transpose directly
    __host__ __device__ Normal3f Transform::applyInverse(Normal3f v) const { return mulTranspose(m, v); } // !!
    __host__ __device__ Point3fi Transform::applyInverse(Point3fi v) const { return mul(mInv, v); }

    __host__ __device__ Ray Transform::applyInverse(Ray const& ray, float* optInOut_tMax) const
    {
        Ray      ret = ray; // NRVO
        Point3fi o   = applyInverse(Point3fi{ray.o});
        ret.d        = normalize(applyInverse(ray.d));
        adjustRangeToErrorBounds(ret.d, o, optInOut_tMax);
        ret.d_inv = 1.f / ret.d; // TODO better

        ret.o = o.midpoint();
        return ret;
    }

    __host__ __device__ RayDifferential Transform::applyInverse(RayDifferential const& ray, float* optInOut_tMax) const
    {
        RayDifferential ret = ray; // NRVO
        Point3fi        o   = applyInverse(Point3fi{ray.o});
        ret.d               = normalize(applyInverse(ret.d));
        adjustRangeToErrorBounds(ret.d, o, optInOut_tMax);
        ret.o     = o.midpoint();
        ret.d_inv = 1.f / ret.d; // TODO better
        // TODO: if differentials are not here, is it correct to skip their transformation
        if (ray.hasDifferentials)
        {
            ret.rxOrigin    = applyInverse(ray.rxOrigin);
            ret.ryOrigin    = applyInverse(ray.ryOrigin);
            ret.rxDirection = applyInverse(ray.rxDirection);
            ret.ryDirection = applyInverse(ray.ryDirection);
        }
        return ret;
    }

    __host__ __device__ Vector3f Transform::operator()(Vector3f v) const { return mul(m, v); }
    __host__ __device__ Point3f  Transform::operator()(Point3f v) const { return mul(m, v); }
    __host__ __device__ Point3fi Transform::operator()(Point3fi v) const { return mul(m, v); }
    __host__ __device__ Normal3f Transform::operator()(Normal3f v) const { return mulTranspose(mInv, v); } // !!

    __host__ __device__ Bounds3f Transform::operator()(Bounds3f const& b) const
    {
        Bounds3f bRet;
        for (int i = 0; i < 8 /*corners*/; ++i)
        {
            Point3f point = b.corner(static_cast<EBoundsCorner>(i));
            bRet          = bbUnion(bRet, operator()(point));
        }
        return bRet;
    }

    __host__ __device__ Ray Transform::operator()(Ray const& ray, float* optInOut_tMax) const
    {
        Ray      ret = ray; // NRVO
        Point3fi o   = operator()(Point3fi{ray.o});
        ret.d        = operator()(ray.d);
        adjustRangeToErrorBounds(ret.d, o, optInOut_tMax);

        ret.o     = o.midpoint();
        ret.d     = normalize(ret.d); // Maybe not necessary?
        ret.d_inv = 1.f / ret.d;      // TODO better
        return ret;
    }

    __host__ __device__ RayDifferential Transform::operator()(RayDifferential const& ray, float* optInOut_tMax) const
    {
        RayDifferential ret = ray; // NRVO
        Point3fi        o   = operator()(Point3fi{ray.o});
        ret.d               = operator()(ret.d);
        adjustRangeToErrorBounds(ret.d, o, optInOut_tMax);
        ret.d_inv = 1.f / ret.d; // TODO better
        ret.o     = o.midpoint();
        // TODO: if differentials are not here, is it correct to skip their transformation
        if (ray.hasDifferentials)
        {
            ret.rxOrigin    = operator()(ray.rxOrigin);
            ret.ryOrigin    = operator()(ray.ryOrigin);
            ret.rxDirection = operator()(ray.rxDirection);
            ret.ryDirection = operator()(ray.ryDirection);
        }
        return ret;
    }

    __host__ __device__ bool Transform::swapsHandedness() const
    {
        glm::mat3 const linearPart{*toGLMmat(&m)};
        return glm::determinant(linearPart) < 0.f;
    }


    DMT_CPU_GPU Transform Translate(Vector3f delta)
    {
        Matrix4f m{{1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0, 1}};
        m = transpose(m);

        return Transform(m);
    }

    // AnimatedTransform ----------------------------------------------------------------------------------------------
    __host__ __device__ AnimatedTransform::DerivativeTerm::DerivativeTerm() : kc(0.f), kx(0.f), ky(0.f), kz(0.f) {}

    __host__ __device__ AnimatedTransform::DerivativeTerm::DerivativeTerm(float c, float x, float y, float z) :
    kc(c),
    kx(x),
    ky(y),
    kz(z)
    {
    }

    __host__ __device__ AnimatedTransform::AnimatedTransform(Transform const& startTransform,
                                                             float            startTime,
                                                             Transform const& endTransform,
                                                             float            endTime) :
    startTransform(startTransform),
    endTransform(endTransform),
    startTime(startTime),
    endTime(endTime),
    m_state(near(startTransform.m, endTransform.m) ? eAnimated : eNone)
    {
        if ((m_state & eAnimated) == 0)
            return;
        // Decompose start and end transformations
        startTransform.decompose(m_t[0], m_r[0], m_s[0]);
        endTransform.decompose(m_t[1], m_r[1], m_s[1]);

        // Flip _R[1]_ if needed to select shortest path
        if (dot(m_r[0], m_r[1]) < 0)
            m_r[1] = -m_r[1];

        if (dot(m_r[0], m_r[1]) < 0.9995f)
            m_state = static_cast<EState>(m_state | eHasRotation);
        // Compute terms of motion derivative function
        if ((m_state & eHasRotation) != 0)
        {
            float cosTheta = dot(m_r[0], m_r[1]);
            float theta    = glm::acos(glm::clamp(cosTheta, -1.f, 1.f));

            Quaternion qperp = normalize(m_r[1] - m_r[0] * cosTheta);

            float t0x    = m_t[0].x;
            float t0y    = m_t[0].y;
            float t0z    = m_t[0].z;
            float t1x    = m_t[1].x;
            float t1y    = m_t[1].y;
            float t1z    = m_t[1].z;
            float q0x    = m_r[0].x;
            float q0y    = m_r[0].y;
            float q0z    = m_r[0].z;
            float q0w    = m_r[0].w;
            float qperpx = qperp.x;
            float qperpy = qperp.y;
            float qperpz = qperp.z;
            float qperpw = qperp.w;
            float s000   = m_s[0][0][0];
            float s001   = m_s[0][0][1];
            float s002   = m_s[0][0][2];
            float s010   = m_s[0][1][0];
            float s011   = m_s[0][1][1];
            float s012   = m_s[0][1][2];
            float s020   = m_s[0][2][0];
            float s021   = m_s[0][2][1];
            float s022   = m_s[0][2][2];
            float s100   = m_s[1][0][0];
            float s101   = m_s[1][0][1];
            float s102   = m_s[1][0][2];
            float s110   = m_s[1][1][0];
            float s111   = m_s[1][1][1];
            float s112   = m_s[1][1][2];
            float s120   = m_s[1][2][0];
            float s121   = m_s[1][2][1];
            float s122   = m_s[1][2][2];

            m_c1[0] = DerivativeTerm(-t0x + t1x,
                                     (-1 + q0y * q0y + q0z * q0z + qperpy * qperpy + qperpz * qperpz) * s000 +
                                         q0w * q0z * s010 - qperpx * qperpy * s010 + qperpw * qperpz * s010 -
                                         q0w * q0y * s020 - qperpw * qperpy * s020 - qperpx * qperpz * s020 + s100 -
                                         q0y * q0y * s100 - q0z * q0z * s100 - qperpy * qperpy * s100 - qperpz * qperpz * s100 -
                                         q0w * q0z * s110 + qperpx * qperpy * s110 - qperpw * qperpz * s110 +
                                         q0w * q0y * s120 + qperpw * qperpy * s120 + qperpx * qperpz * s120 +
                                         q0x * (-(q0y * s010) - q0z * s020 + q0y * s110 + q0z * s120),
                                     (-1 + q0y * q0y + q0z * q0z + qperpy * qperpy + qperpz * qperpz) * s001 +
                                         q0w * q0z * s011 - qperpx * qperpy * s011 + qperpw * qperpz * s011 -
                                         q0w * q0y * s021 - qperpw * qperpy * s021 - qperpx * qperpz * s021 + s101 -
                                         q0y * q0y * s101 - q0z * q0z * s101 - qperpy * qperpy * s101 - qperpz * qperpz * s101 -
                                         q0w * q0z * s111 + qperpx * qperpy * s111 - qperpw * qperpz * s111 +
                                         q0w * q0y * s121 + qperpw * qperpy * s121 + qperpx * qperpz * s121 +
                                         q0x * (-(q0y * s011) - q0z * s021 + q0y * s111 + q0z * s121),
                                     (-1 + q0y * q0y + q0z * q0z + qperpy * qperpy + qperpz * qperpz) * s002 +
                                         q0w * q0z * s012 - qperpx * qperpy * s012 + qperpw * qperpz * s012 -
                                         q0w * q0y * s022 - qperpw * qperpy * s022 - qperpx * qperpz * s022 + s102 -
                                         q0y * q0y * s102 - q0z * q0z * s102 - qperpy * qperpy * s102 - qperpz * qperpz * s102 -
                                         q0w * q0z * s112 + qperpx * qperpy * s112 - qperpw * qperpz * s112 +
                                         q0w * q0y * s122 + qperpw * qperpy * s122 + qperpx * qperpz * s122 +
                                         q0x * (-(q0y * s012) - q0z * s022 + q0y * s112 + q0z * s122));

            m_c2[0] = DerivativeTerm(0.,
                                     -(qperpy * qperpy * s000) - qperpz * qperpz * s000 + qperpx * qperpy * s010 -
                                         qperpw * qperpz * s010 + qperpw * qperpy * s020 + qperpx * qperpz * s020 +
                                         q0y * q0y * (s000 - s100) + q0z * q0z * (s000 - s100) +
                                         qperpy * qperpy * s100 + qperpz * qperpz * s100 - qperpx * qperpy * s110 +
                                         qperpw * qperpz * s110 - qperpw * qperpy * s120 - qperpx * qperpz * s120 +
                                         2 * q0x * qperpy * s010 * theta - 2 * q0w * qperpz * s010 * theta +
                                         2 * q0w * qperpy * s020 * theta + 2 * q0x * qperpz * s020 * theta +
                                         q0y * (q0x * (-s010 + s110) + q0w * (-s020 + s120) +
                                                2 * (-2 * qperpy * s000 + qperpx * s010 + qperpw * s020) * theta) +
                                         q0z * (q0w * (s010 - s110) + q0x * (-s020 + s120) -
                                                2 * (2 * qperpz * s000 + qperpw * s010 - qperpx * s020) * theta),
                                     -(qperpy * qperpy * s001) - qperpz * qperpz * s001 + qperpx * qperpy * s011 -
                                         qperpw * qperpz * s011 + qperpw * qperpy * s021 + qperpx * qperpz * s021 +
                                         q0y * q0y * (s001 - s101) + q0z * q0z * (s001 - s101) +
                                         qperpy * qperpy * s101 + qperpz * qperpz * s101 - qperpx * qperpy * s111 +
                                         qperpw * qperpz * s111 - qperpw * qperpy * s121 - qperpx * qperpz * s121 +
                                         2 * q0x * qperpy * s011 * theta - 2 * q0w * qperpz * s011 * theta +
                                         2 * q0w * qperpy * s021 * theta + 2 * q0x * qperpz * s021 * theta +
                                         q0y * (q0x * (-s011 + s111) + q0w * (-s021 + s121) +
                                                2 * (-2 * qperpy * s001 + qperpx * s011 + qperpw * s021) * theta) +
                                         q0z * (q0w * (s011 - s111) + q0x * (-s021 + s121) -
                                                2 * (2 * qperpz * s001 + qperpw * s011 - qperpx * s021) * theta),
                                     -(qperpy * qperpy * s002) - qperpz * qperpz * s002 + qperpx * qperpy * s012 -
                                         qperpw * qperpz * s012 + qperpw * qperpy * s022 + qperpx * qperpz * s022 +
                                         q0y * q0y * (s002 - s102) + q0z * q0z * (s002 - s102) +
                                         qperpy * qperpy * s102 + qperpz * qperpz * s102 - qperpx * qperpy * s112 +
                                         qperpw * qperpz * s112 - qperpw * qperpy * s122 - qperpx * qperpz * s122 +
                                         2 * q0x * qperpy * s012 * theta - 2 * q0w * qperpz * s012 * theta +
                                         2 * q0w * qperpy * s022 * theta + 2 * q0x * qperpz * s022 * theta +
                                         q0y * (q0x * (-s012 + s112) + q0w * (-s022 + s122) +
                                                2 * (-2 * qperpy * s002 + qperpx * s012 + qperpw * s022) * theta) +
                                         q0z * (q0w * (s012 - s112) + q0x * (-s022 + s122) -
                                                2 * (2 * qperpz * s002 + qperpw * s012 - qperpx * s022) * theta));

            m_c3[0] = DerivativeTerm(0.,
                                     -2 *
                                         (q0x * qperpy * s010 - q0w * qperpz * s010 + q0w * qperpy * s020 +
                                          q0x * qperpz * s020 - q0x * qperpy * s110 + q0w * qperpz * s110 -
                                          q0w * qperpy * s120 - q0x * qperpz * s120 +
                                          q0y * (-2 * qperpy * s000 + qperpx * s010 + qperpw * s020 +
                                                 2 * qperpy * s100 - qperpx * s110 - qperpw * s120) +
                                          q0z * (-2 * qperpz * s000 - qperpw * s010 + qperpx * s020 +
                                                 2 * qperpz * s100 + qperpw * s110 - qperpx * s120)) *
                                         theta,
                                     -2 *
                                         (q0x * qperpy * s011 - q0w * qperpz * s011 + q0w * qperpy * s021 +
                                          q0x * qperpz * s021 - q0x * qperpy * s111 + q0w * qperpz * s111 -
                                          q0w * qperpy * s121 - q0x * qperpz * s121 +
                                          q0y * (-2 * qperpy * s001 + qperpx * s011 + qperpw * s021 +
                                                 2 * qperpy * s101 - qperpx * s111 - qperpw * s121) +
                                          q0z * (-2 * qperpz * s001 - qperpw * s011 + qperpx * s021 +
                                                 2 * qperpz * s101 + qperpw * s111 - qperpx * s121)) *
                                         theta,
                                     -2 *
                                         (q0x * qperpy * s012 - q0w * qperpz * s012 + q0w * qperpy * s022 +
                                          q0x * qperpz * s022 - q0x * qperpy * s112 + q0w * qperpz * s112 -
                                          q0w * qperpy * s122 - q0x * qperpz * s122 +
                                          q0y * (-2 * qperpy * s002 + qperpx * s012 + qperpw * s022 +
                                                 2 * qperpy * s102 - qperpx * s112 - qperpw * s122) +
                                          q0z * (-2 * qperpz * s002 - qperpw * s012 + qperpx * s022 +
                                                 2 * qperpz * s102 + qperpw * s112 - qperpx * s122)) *
                                         theta);

            m_c4[0] = DerivativeTerm(
                0.,
                -(q0x * qperpy * s010) + q0w * qperpz * s010 - q0w * qperpy * s020 - q0x * qperpz * s020 +
                    q0x * qperpy * s110 - q0w * qperpz * s110 + q0w * qperpy * s120 + q0x * qperpz * s120 +
                    2 * q0y * q0y * s000 * theta + 2 * q0z * q0z * s000 * theta - 2 * qperpy * qperpy * s000 * theta -
                    2 * qperpz * qperpz * s000 * theta + 2 * qperpx * qperpy * s010 * theta -
                    2 * qperpw * qperpz * s010 * theta + 2 * qperpw * qperpy * s020 * theta +
                    2 * qperpx * qperpz * s020 * theta +
                    q0y * (-(qperpx * s010) - qperpw * s020 + 2 * qperpy * (s000 - s100) + qperpx * s110 +
                           qperpw * s120 - 2 * q0x * s010 * theta - 2 * q0w * s020 * theta) +
                    q0z * (2 * qperpz * s000 + qperpw * s010 - qperpx * s020 - 2 * qperpz * s100 - qperpw * s110 +
                           qperpx * s120 + 2 * q0w * s010 * theta - 2 * q0x * s020 * theta),
                -(q0x * qperpy * s011) + q0w * qperpz * s011 - q0w * qperpy * s021 - q0x * qperpz * s021 +
                    q0x * qperpy * s111 - q0w * qperpz * s111 + q0w * qperpy * s121 + q0x * qperpz * s121 +
                    2 * q0y * q0y * s001 * theta + 2 * q0z * q0z * s001 * theta - 2 * qperpy * qperpy * s001 * theta -
                    2 * qperpz * qperpz * s001 * theta + 2 * qperpx * qperpy * s011 * theta -
                    2 * qperpw * qperpz * s011 * theta + 2 * qperpw * qperpy * s021 * theta +
                    2 * qperpx * qperpz * s021 * theta +
                    q0y * (-(qperpx * s011) - qperpw * s021 + 2 * qperpy * (s001 - s101) + qperpx * s111 +
                           qperpw * s121 - 2 * q0x * s011 * theta - 2 * q0w * s021 * theta) +
                    q0z * (2 * qperpz * s001 + qperpw * s011 - qperpx * s021 - 2 * qperpz * s101 - qperpw * s111 +
                           qperpx * s121 + 2 * q0w * s011 * theta - 2 * q0x * s021 * theta),
                -(q0x * qperpy * s012) + q0w * qperpz * s012 - q0w * qperpy * s022 - q0x * qperpz * s022 +
                    q0x * qperpy * s112 - q0w * qperpz * s112 + q0w * qperpy * s122 + q0x * qperpz * s122 +
                    2 * q0y * q0y * s002 * theta + 2 * q0z * q0z * s002 * theta - 2 * qperpy * qperpy * s002 * theta -
                    2 * qperpz * qperpz * s002 * theta + 2 * qperpx * qperpy * s012 * theta -
                    2 * qperpw * qperpz * s012 * theta + 2 * qperpw * qperpy * s022 * theta +
                    2 * qperpx * qperpz * s022 * theta +
                    q0y * (-(qperpx * s012) - qperpw * s022 + 2 * qperpy * (s002 - s102) + qperpx * s112 +
                           qperpw * s122 - 2 * q0x * s012 * theta - 2 * q0w * s022 * theta) +
                    q0z * (2 * qperpz * s002 + qperpw * s012 - qperpx * s022 - 2 * qperpz * s102 - qperpw * s112 +
                           qperpx * s122 + 2 * q0w * s012 * theta - 2 * q0x * s022 * theta));

            m_c5[0] = DerivativeTerm(0.,
                                     2 *
                                         (qperpy * qperpy * s000 + qperpz * qperpz * s000 - qperpx * qperpy * s010 +
                                          qperpw * qperpz * s010 - qperpw * qperpy * s020 - qperpx * qperpz * s020 -
                                          qperpy * qperpy * s100 - qperpz * qperpz * s100 + q0y * q0y * (-s000 + s100) +
                                          q0z * q0z * (-s000 + s100) + qperpx * qperpy * s110 - qperpw * qperpz * s110 +
                                          q0y * (q0x * (s010 - s110) + q0w * (s020 - s120)) + qperpw * qperpy * s120 +
                                          qperpx * qperpz * s120 +
                                          q0z * (-(q0w * s010) + q0x * s020 + q0w * s110 - q0x * s120)) *
                                         theta,
                                     2 *
                                         (qperpy * qperpy * s001 + qperpz * qperpz * s001 - qperpx * qperpy * s011 +
                                          qperpw * qperpz * s011 - qperpw * qperpy * s021 - qperpx * qperpz * s021 -
                                          qperpy * qperpy * s101 - qperpz * qperpz * s101 + q0y * q0y * (-s001 + s101) +
                                          q0z * q0z * (-s001 + s101) + qperpx * qperpy * s111 - qperpw * qperpz * s111 +
                                          q0y * (q0x * (s011 - s111) + q0w * (s021 - s121)) + qperpw * qperpy * s121 +
                                          qperpx * qperpz * s121 +
                                          q0z * (-(q0w * s011) + q0x * s021 + q0w * s111 - q0x * s121)) *
                                         theta,
                                     2 *
                                         (qperpy * qperpy * s002 + qperpz * qperpz * s002 - qperpx * qperpy * s012 +
                                          qperpw * qperpz * s012 - qperpw * qperpy * s022 - qperpx * qperpz * s022 -
                                          qperpy * qperpy * s102 - qperpz * qperpz * s102 + q0y * q0y * (-s002 + s102) +
                                          q0z * q0z * (-s002 + s102) + qperpx * qperpy * s112 - qperpw * qperpz * s112 +
                                          q0y * (q0x * (s012 - s112) + q0w * (s022 - s122)) + qperpw * qperpy * s122 +
                                          qperpx * qperpz * s122 +
                                          q0z * (-(q0w * s012) + q0x * s022 + q0w * s112 - q0x * s122)) *
                                         theta);

            m_c1[1] = DerivativeTerm(-t0y + t1y,
                                     -(qperpx * qperpy * s000) - qperpw * qperpz * s000 - s010 + q0z * q0z * s010 +
                                         qperpx * qperpx * s010 + qperpz * qperpz * s010 - q0y * q0z * s020 +
                                         qperpw * qperpx * s020 - qperpy * qperpz * s020 + qperpx * qperpy * s100 +
                                         qperpw * qperpz * s100 + q0w * q0z * (-s000 + s100) +
                                         q0x * q0x * (s010 - s110) + s110 - q0z * q0z * s110 - qperpx * qperpx * s110 -
                                         qperpz * qperpz * s110 + q0x * (q0y * (-s000 + s100) + q0w * (s020 - s120)) +
                                         q0y * q0z * s120 - qperpw * qperpx * s120 + qperpy * qperpz * s120,
                                     -(qperpx * qperpy * s001) - qperpw * qperpz * s001 - s011 + q0z * q0z * s011 +
                                         qperpx * qperpx * s011 + qperpz * qperpz * s011 - q0y * q0z * s021 +
                                         qperpw * qperpx * s021 - qperpy * qperpz * s021 + qperpx * qperpy * s101 +
                                         qperpw * qperpz * s101 + q0w * q0z * (-s001 + s101) +
                                         q0x * q0x * (s011 - s111) + s111 - q0z * q0z * s111 - qperpx * qperpx * s111 -
                                         qperpz * qperpz * s111 + q0x * (q0y * (-s001 + s101) + q0w * (s021 - s121)) +
                                         q0y * q0z * s121 - qperpw * qperpx * s121 + qperpy * qperpz * s121,
                                     -(qperpx * qperpy * s002) - qperpw * qperpz * s002 - s012 + q0z * q0z * s012 +
                                         qperpx * qperpx * s012 + qperpz * qperpz * s012 - q0y * q0z * s022 +
                                         qperpw * qperpx * s022 - qperpy * qperpz * s022 + qperpx * qperpy * s102 +
                                         qperpw * qperpz * s102 + q0w * q0z * (-s002 + s102) +
                                         q0x * q0x * (s012 - s112) + s112 - q0z * q0z * s112 - qperpx * qperpx * s112 -
                                         qperpz * qperpz * s112 + q0x * (q0y * (-s002 + s102) + q0w * (s022 - s122)) +
                                         q0y * q0z * s122 - qperpw * qperpx * s122 + qperpy * qperpz * s122);

            m_c2[1] = DerivativeTerm(
                0.,
                qperpx * qperpy * s000 + qperpw * qperpz * s000 + q0z * q0z * s010 - qperpx * qperpx * s010 -
                    qperpz * qperpz * s010 - q0y * q0z * s020 - qperpw * qperpx * s020 + qperpy * qperpz * s020 -
                    qperpx * qperpy * s100 - qperpw * qperpz * s100 + q0x * q0x * (s010 - s110) - q0z * q0z * s110 +
                    qperpx * qperpx * s110 + qperpz * qperpz * s110 + q0y * q0z * s120 + qperpw * qperpx * s120 -
                    qperpy * qperpz * s120 + 2 * q0z * qperpw * s000 * theta + 2 * q0y * qperpx * s000 * theta -
                    4 * q0z * qperpz * s010 * theta + 2 * q0z * qperpy * s020 * theta + 2 * q0y * qperpz * s020 * theta +
                    q0x * (q0w * s020 + q0y * (-s000 + s100) - q0w * s120 + 2 * qperpy * s000 * theta -
                           4 * qperpx * s010 * theta - 2 * qperpw * s020 * theta) +
                    q0w * (-(q0z * s000) + q0z * s100 + 2 * qperpz * s000 * theta - 2 * qperpx * s020 * theta),
                qperpx * qperpy * s001 + qperpw * qperpz * s001 + q0z * q0z * s011 - qperpx * qperpx * s011 -
                    qperpz * qperpz * s011 - q0y * q0z * s021 - qperpw * qperpx * s021 + qperpy * qperpz * s021 -
                    qperpx * qperpy * s101 - qperpw * qperpz * s101 + q0x * q0x * (s011 - s111) - q0z * q0z * s111 +
                    qperpx * qperpx * s111 + qperpz * qperpz * s111 + q0y * q0z * s121 + qperpw * qperpx * s121 -
                    qperpy * qperpz * s121 + 2 * q0z * qperpw * s001 * theta + 2 * q0y * qperpx * s001 * theta -
                    4 * q0z * qperpz * s011 * theta + 2 * q0z * qperpy * s021 * theta + 2 * q0y * qperpz * s021 * theta +
                    q0x * (q0w * s021 + q0y * (-s001 + s101) - q0w * s121 + 2 * qperpy * s001 * theta -
                           4 * qperpx * s011 * theta - 2 * qperpw * s021 * theta) +
                    q0w * (-(q0z * s001) + q0z * s101 + 2 * qperpz * s001 * theta - 2 * qperpx * s021 * theta),
                qperpx * qperpy * s002 + qperpw * qperpz * s002 + q0z * q0z * s012 - qperpx * qperpx * s012 -
                    qperpz * qperpz * s012 - q0y * q0z * s022 - qperpw * qperpx * s022 + qperpy * qperpz * s022 -
                    qperpx * qperpy * s102 - qperpw * qperpz * s102 + q0x * q0x * (s012 - s112) - q0z * q0z * s112 +
                    qperpx * qperpx * s112 + qperpz * qperpz * s112 + q0y * q0z * s122 + qperpw * qperpx * s122 -
                    qperpy * qperpz * s122 + 2 * q0z * qperpw * s002 * theta + 2 * q0y * qperpx * s002 * theta -
                    4 * q0z * qperpz * s012 * theta + 2 * q0z * qperpy * s022 * theta + 2 * q0y * qperpz * s022 * theta +
                    q0x * (q0w * s022 + q0y * (-s002 + s102) - q0w * s122 + 2 * qperpy * s002 * theta -
                           4 * qperpx * s012 * theta - 2 * qperpw * s022 * theta) +
                    q0w * (-(q0z * s002) + q0z * s102 + 2 * qperpz * s002 * theta - 2 * qperpx * s022 * theta));

            m_c3[1] = DerivativeTerm(0.,
                                     2 *
                                         (-(q0x * qperpy * s000) - q0w * qperpz * s000 + 2 * q0x * qperpx * s010 +
                                          q0x * qperpw * s020 + q0w * qperpx * s020 + q0x * qperpy * s100 + q0w * qperpz * s100 -
                                          2 * q0x * qperpx * s110 - q0x * qperpw * s120 - q0w * qperpx * s120 +
                                          q0z * (2 * qperpz * s010 - qperpy * s020 + qperpw * (-s000 + s100) -
                                                 2 * qperpz * s110 + qperpy * s120) +
                                          q0y * (-(qperpx * s000) - qperpz * s020 + qperpx * s100 + qperpz * s120)) *
                                         theta,
                                     2 *
                                         (-(q0x * qperpy * s001) - q0w * qperpz * s001 + 2 * q0x * qperpx * s011 +
                                          q0x * qperpw * s021 + q0w * qperpx * s021 + q0x * qperpy * s101 + q0w * qperpz * s101 -
                                          2 * q0x * qperpx * s111 - q0x * qperpw * s121 - q0w * qperpx * s121 +
                                          q0z * (2 * qperpz * s011 - qperpy * s021 + qperpw * (-s001 + s101) -
                                                 2 * qperpz * s111 + qperpy * s121) +
                                          q0y * (-(qperpx * s001) - qperpz * s021 + qperpx * s101 + qperpz * s121)) *
                                         theta,
                                     2 *
                                         (-(q0x * qperpy * s002) - q0w * qperpz * s002 + 2 * q0x * qperpx * s012 +
                                          q0x * qperpw * s022 + q0w * qperpx * s022 + q0x * qperpy * s102 + q0w * qperpz * s102 -
                                          2 * q0x * qperpx * s112 - q0x * qperpw * s122 - q0w * qperpx * s122 +
                                          q0z * (2 * qperpz * s012 - qperpy * s022 + qperpw * (-s002 + s102) -
                                                 2 * qperpz * s112 + qperpy * s122) +
                                          q0y * (-(qperpx * s002) - qperpz * s022 + qperpx * s102 + qperpz * s122)) *
                                         theta);

            m_c4[1] = DerivativeTerm(
                0.,
                -(q0x * qperpy * s000) - q0w * qperpz * s000 + 2 * q0x * qperpx * s010 + q0x * qperpw * s020 +
                    q0w * qperpx * s020 + q0x * qperpy * s100 + q0w * qperpz * s100 - 2 * q0x * qperpx * s110 -
                    q0x * qperpw * s120 - q0w * qperpx * s120 + 2 * qperpx * qperpy * s000 * theta +
                    2 * qperpw * qperpz * s000 * theta + 2 * q0x * q0x * s010 * theta + 2 * q0z * q0z * s010 * theta -
                    2 * qperpx * qperpx * s010 * theta - 2 * qperpz * qperpz * s010 * theta +
                    2 * q0w * q0x * s020 * theta - 2 * qperpw * qperpx * s020 * theta + 2 * qperpy * qperpz * s020 * theta +
                    q0y * (-(qperpx * s000) - qperpz * s020 + qperpx * s100 + qperpz * s120 - 2 * q0x * s000 * theta) +
                    q0z * (2 * qperpz * s010 - qperpy * s020 + qperpw * (-s000 + s100) - 2 * qperpz * s110 +
                           qperpy * s120 - 2 * q0w * s000 * theta - 2 * q0y * s020 * theta),
                -(q0x * qperpy * s001) - q0w * qperpz * s001 + 2 * q0x * qperpx * s011 + q0x * qperpw * s021 +
                    q0w * qperpx * s021 + q0x * qperpy * s101 + q0w * qperpz * s101 - 2 * q0x * qperpx * s111 -
                    q0x * qperpw * s121 - q0w * qperpx * s121 + 2 * qperpx * qperpy * s001 * theta +
                    2 * qperpw * qperpz * s001 * theta + 2 * q0x * q0x * s011 * theta + 2 * q0z * q0z * s011 * theta -
                    2 * qperpx * qperpx * s011 * theta - 2 * qperpz * qperpz * s011 * theta +
                    2 * q0w * q0x * s021 * theta - 2 * qperpw * qperpx * s021 * theta + 2 * qperpy * qperpz * s021 * theta +
                    q0y * (-(qperpx * s001) - qperpz * s021 + qperpx * s101 + qperpz * s121 - 2 * q0x * s001 * theta) +
                    q0z * (2 * qperpz * s011 - qperpy * s021 + qperpw * (-s001 + s101) - 2 * qperpz * s111 +
                           qperpy * s121 - 2 * q0w * s001 * theta - 2 * q0y * s021 * theta),
                -(q0x * qperpy * s002) - q0w * qperpz * s002 + 2 * q0x * qperpx * s012 + q0x * qperpw * s022 +
                    q0w * qperpx * s022 + q0x * qperpy * s102 + q0w * qperpz * s102 - 2 * q0x * qperpx * s112 -
                    q0x * qperpw * s122 - q0w * qperpx * s122 + 2 * qperpx * qperpy * s002 * theta +
                    2 * qperpw * qperpz * s002 * theta + 2 * q0x * q0x * s012 * theta + 2 * q0z * q0z * s012 * theta -
                    2 * qperpx * qperpx * s012 * theta - 2 * qperpz * qperpz * s012 * theta +
                    2 * q0w * q0x * s022 * theta - 2 * qperpw * qperpx * s022 * theta + 2 * qperpy * qperpz * s022 * theta +
                    q0y * (-(qperpx * s002) - qperpz * s022 + qperpx * s102 + qperpz * s122 - 2 * q0x * s002 * theta) +
                    q0z * (2 * qperpz * s012 - qperpy * s022 + qperpw * (-s002 + s102) - 2 * qperpz * s112 +
                           qperpy * s122 - 2 * q0w * s002 * theta - 2 * q0y * s022 * theta));

            m_c5[1] = DerivativeTerm(0.,
                                     -2 *
                                         (qperpx * qperpy * s000 + qperpw * qperpz * s000 + q0z * q0z * s010 -
                                          qperpx * qperpx * s010 - qperpz * qperpz * s010 - q0y * q0z * s020 -
                                          qperpw * qperpx * s020 + qperpy * qperpz * s020 - qperpx * qperpy * s100 -
                                          qperpw * qperpz * s100 + q0w * q0z * (-s000 + s100) +
                                          q0x * q0x * (s010 - s110) - q0z * q0z * s110 + qperpx * qperpx * s110 +
                                          qperpz * qperpz * s110 + q0x * (q0y * (-s000 + s100) + q0w * (s020 - s120)) +
                                          q0y * q0z * s120 + qperpw * qperpx * s120 - qperpy * qperpz * s120) *
                                         theta,
                                     -2 *
                                         (qperpx * qperpy * s001 + qperpw * qperpz * s001 + q0z * q0z * s011 -
                                          qperpx * qperpx * s011 - qperpz * qperpz * s011 - q0y * q0z * s021 -
                                          qperpw * qperpx * s021 + qperpy * qperpz * s021 - qperpx * qperpy * s101 -
                                          qperpw * qperpz * s101 + q0w * q0z * (-s001 + s101) +
                                          q0x * q0x * (s011 - s111) - q0z * q0z * s111 + qperpx * qperpx * s111 +
                                          qperpz * qperpz * s111 + q0x * (q0y * (-s001 + s101) + q0w * (s021 - s121)) +
                                          q0y * q0z * s121 + qperpw * qperpx * s121 - qperpy * qperpz * s121) *
                                         theta,
                                     -2 *
                                         (qperpx * qperpy * s002 + qperpw * qperpz * s002 + q0z * q0z * s012 -
                                          qperpx * qperpx * s012 - qperpz * qperpz * s012 - q0y * q0z * s022 -
                                          qperpw * qperpx * s022 + qperpy * qperpz * s022 - qperpx * qperpy * s102 -
                                          qperpw * qperpz * s102 + q0w * q0z * (-s002 + s102) +
                                          q0x * q0x * (s012 - s112) - q0z * q0z * s112 + qperpx * qperpx * s112 +
                                          qperpz * qperpz * s112 + q0x * (q0y * (-s002 + s102) + q0w * (s022 - s122)) +
                                          q0y * q0z * s122 + qperpw * qperpx * s122 - qperpy * qperpz * s122) *
                                         theta);

            m_c1[2] = DerivativeTerm(-t0z + t1z,
                                     (qperpw * qperpy * s000 - qperpx * qperpz * s000 - q0y * q0z * s010 -
                                      qperpw * qperpx * s010 - qperpy * qperpz * s010 - s020 + q0y * q0y * s020 +
                                      qperpx * qperpx * s020 + qperpy * qperpy * s020 - qperpw * qperpy * s100 +
                                      qperpx * qperpz * s100 + q0x * q0z * (-s000 + s100) + q0y * q0z * s110 +
                                      qperpw * qperpx * s110 + qperpy * qperpz * s110 +
                                      q0w * (q0y * (s000 - s100) + q0x * (-s010 + s110)) + q0x * q0x * (s020 - s120) +
                                      s120 - q0y * q0y * s120 - qperpx * qperpx * s120 - qperpy * qperpy * s120),
                                     (qperpw * qperpy * s001 - qperpx * qperpz * s001 - q0y * q0z * s011 -
                                      qperpw * qperpx * s011 - qperpy * qperpz * s011 - s021 + q0y * q0y * s021 +
                                      qperpx * qperpx * s021 + qperpy * qperpy * s021 - qperpw * qperpy * s101 +
                                      qperpx * qperpz * s101 + q0x * q0z * (-s001 + s101) + q0y * q0z * s111 +
                                      qperpw * qperpx * s111 + qperpy * qperpz * s111 +
                                      q0w * (q0y * (s001 - s101) + q0x * (-s011 + s111)) + q0x * q0x * (s021 - s121) +
                                      s121 - q0y * q0y * s121 - qperpx * qperpx * s121 - qperpy * qperpy * s121),
                                     (qperpw * qperpy * s002 - qperpx * qperpz * s002 - q0y * q0z * s012 -
                                      qperpw * qperpx * s012 - qperpy * qperpz * s012 - s022 + q0y * q0y * s022 +
                                      qperpx * qperpx * s022 + qperpy * qperpy * s022 - qperpw * qperpy * s102 +
                                      qperpx * qperpz * s102 + q0x * q0z * (-s002 + s102) + q0y * q0z * s112 +
                                      qperpw * qperpx * s112 + qperpy * qperpz * s112 +
                                      q0w * (q0y * (s002 - s102) + q0x * (-s012 + s112)) + q0x * q0x * (s022 - s122) +
                                      s122 - q0y * q0y * s122 - qperpx * qperpx * s122 - qperpy * qperpy * s122));

            m_c2[2] = DerivativeTerm(
                0.,
                (q0w * q0y * s000 - q0x * q0z * s000 - qperpw * qperpy * s000 + qperpx * qperpz * s000 -
                 q0w * q0x * s010 - q0y * q0z * s010 + qperpw * qperpx * s010 + qperpy * qperpz * s010 +
                 q0x * q0x * s020 + q0y * q0y * s020 - qperpx * qperpx * s020 - qperpy * qperpy * s020 -
                 q0w * q0y * s100 + q0x * q0z * s100 + qperpw * qperpy * s100 - qperpx * qperpz * s100 +
                 q0w * q0x * s110 + q0y * q0z * s110 - qperpw * qperpx * s110 - qperpy * qperpz * s110 - q0x * q0x * s120 -
                 q0y * q0y * s120 + qperpx * qperpx * s120 + qperpy * qperpy * s120 - 2 * q0y * qperpw * s000 * theta +
                 2 * q0z * qperpx * s000 * theta - 2 * q0w * qperpy * s000 * theta + 2 * q0x * qperpz * s000 * theta +
                 2 * q0x * qperpw * s010 * theta + 2 * q0w * qperpx * s010 * theta + 2 * q0z * qperpy * s010 * theta +
                 2 * q0y * qperpz * s010 * theta - 4 * q0x * qperpx * s020 * theta - 4 * q0y * qperpy * s020 * theta),
                (q0w * q0y * s001 - q0x * q0z * s001 - qperpw * qperpy * s001 + qperpx * qperpz * s001 -
                 q0w * q0x * s011 - q0y * q0z * s011 + qperpw * qperpx * s011 + qperpy * qperpz * s011 +
                 q0x * q0x * s021 + q0y * q0y * s021 - qperpx * qperpx * s021 - qperpy * qperpy * s021 -
                 q0w * q0y * s101 + q0x * q0z * s101 + qperpw * qperpy * s101 - qperpx * qperpz * s101 +
                 q0w * q0x * s111 + q0y * q0z * s111 - qperpw * qperpx * s111 - qperpy * qperpz * s111 - q0x * q0x * s121 -
                 q0y * q0y * s121 + qperpx * qperpx * s121 + qperpy * qperpy * s121 - 2 * q0y * qperpw * s001 * theta +
                 2 * q0z * qperpx * s001 * theta - 2 * q0w * qperpy * s001 * theta + 2 * q0x * qperpz * s001 * theta +
                 2 * q0x * qperpw * s011 * theta + 2 * q0w * qperpx * s011 * theta + 2 * q0z * qperpy * s011 * theta +
                 2 * q0y * qperpz * s011 * theta - 4 * q0x * qperpx * s021 * theta - 4 * q0y * qperpy * s021 * theta),
                (q0w * q0y * s002 - q0x * q0z * s002 - qperpw * qperpy * s002 + qperpx * qperpz * s002 -
                 q0w * q0x * s012 - q0y * q0z * s012 + qperpw * qperpx * s012 + qperpy * qperpz * s012 +
                 q0x * q0x * s022 + q0y * q0y * s022 - qperpx * qperpx * s022 - qperpy * qperpy * s022 -
                 q0w * q0y * s102 + q0x * q0z * s102 + qperpw * qperpy * s102 - qperpx * qperpz * s102 +
                 q0w * q0x * s112 + q0y * q0z * s112 - qperpw * qperpx * s112 - qperpy * qperpz * s112 - q0x * q0x * s122 -
                 q0y * q0y * s122 + qperpx * qperpx * s122 + qperpy * qperpy * s122 - 2 * q0y * qperpw * s002 * theta +
                 2 * q0z * qperpx * s002 * theta - 2 * q0w * qperpy * s002 * theta + 2 * q0x * qperpz * s002 * theta +
                 2 * q0x * qperpw * s012 * theta + 2 * q0w * qperpx * s012 * theta + 2 * q0z * qperpy * s012 * theta +
                 2 * q0y * qperpz * s012 * theta - 4 * q0x * qperpx * s022 * theta - 4 * q0y * qperpy * s022 * theta));

            m_c3[2] = DerivativeTerm(0.,
                                     -2 *
                                         (-(q0w * qperpy * s000) + q0x * qperpz * s000 + q0x * qperpw * s010 +
                                          q0w * qperpx * s010 - 2 * q0x * qperpx * s020 + q0w * qperpy * s100 -
                                          q0x * qperpz * s100 - q0x * qperpw * s110 - q0w * qperpx * s110 +
                                          q0z * (qperpx * s000 + qperpy * s010 - qperpx * s100 - qperpy * s110) +
                                          2 * q0x * qperpx * s120 +
                                          q0y * (qperpz * s010 - 2 * qperpy * s020 + qperpw * (-s000 + s100) -
                                                 qperpz * s110 + 2 * qperpy * s120)) *
                                         theta,
                                     -2 *
                                         (-(q0w * qperpy * s001) + q0x * qperpz * s001 + q0x * qperpw * s011 +
                                          q0w * qperpx * s011 - 2 * q0x * qperpx * s021 + q0w * qperpy * s101 -
                                          q0x * qperpz * s101 - q0x * qperpw * s111 - q0w * qperpx * s111 +
                                          q0z * (qperpx * s001 + qperpy * s011 - qperpx * s101 - qperpy * s111) +
                                          2 * q0x * qperpx * s121 +
                                          q0y * (qperpz * s011 - 2 * qperpy * s021 + qperpw * (-s001 + s101) -
                                                 qperpz * s111 + 2 * qperpy * s121)) *
                                         theta,
                                     -2 *
                                         (-(q0w * qperpy * s002) + q0x * qperpz * s002 + q0x * qperpw * s012 +
                                          q0w * qperpx * s012 - 2 * q0x * qperpx * s022 + q0w * qperpy * s102 -
                                          q0x * qperpz * s102 - q0x * qperpw * s112 - q0w * qperpx * s112 +
                                          q0z * (qperpx * s002 + qperpy * s012 - qperpx * s102 - qperpy * s112) +
                                          2 * q0x * qperpx * s122 +
                                          q0y * (qperpz * s012 - 2 * qperpy * s022 + qperpw * (-s002 + s102) -
                                                 qperpz * s112 + 2 * qperpy * s122)) *
                                         theta);

            m_c4[2] = DerivativeTerm(
                0.,
                q0w * qperpy * s000 - q0x * qperpz * s000 - q0x * qperpw * s010 - q0w * qperpx * s010 +
                    2 * q0x * qperpx * s020 - q0w * qperpy * s100 + q0x * qperpz * s100 + q0x * qperpw * s110 +
                    q0w * qperpx * s110 - 2 * q0x * qperpx * s120 - 2 * qperpw * qperpy * s000 * theta +
                    2 * qperpx * qperpz * s000 * theta - 2 * q0w * q0x * s010 * theta + 2 * qperpw * qperpx * s010 * theta +
                    2 * qperpy * qperpz * s010 * theta + 2 * q0x * q0x * s020 * theta + 2 * q0y * q0y * s020 * theta -
                    2 * qperpx * qperpx * s020 * theta - 2 * qperpy * qperpy * s020 * theta +
                    q0z * (-(qperpx * s000) - qperpy * s010 + qperpx * s100 + qperpy * s110 - 2 * q0x * s000 * theta) +
                    q0y * (-(qperpz * s010) + 2 * qperpy * s020 + qperpw * (s000 - s100) + qperpz * s110 -
                           2 * qperpy * s120 + 2 * q0w * s000 * theta - 2 * q0z * s010 * theta),
                q0w * qperpy * s001 - q0x * qperpz * s001 - q0x * qperpw * s011 - q0w * qperpx * s011 +
                    2 * q0x * qperpx * s021 - q0w * qperpy * s101 + q0x * qperpz * s101 + q0x * qperpw * s111 +
                    q0w * qperpx * s111 - 2 * q0x * qperpx * s121 - 2 * qperpw * qperpy * s001 * theta +
                    2 * qperpx * qperpz * s001 * theta - 2 * q0w * q0x * s011 * theta + 2 * qperpw * qperpx * s011 * theta +
                    2 * qperpy * qperpz * s011 * theta + 2 * q0x * q0x * s021 * theta + 2 * q0y * q0y * s021 * theta -
                    2 * qperpx * qperpx * s021 * theta - 2 * qperpy * qperpy * s021 * theta +
                    q0z * (-(qperpx * s001) - qperpy * s011 + qperpx * s101 + qperpy * s111 - 2 * q0x * s001 * theta) +
                    q0y * (-(qperpz * s011) + 2 * qperpy * s021 + qperpw * (s001 - s101) + qperpz * s111 -
                           2 * qperpy * s121 + 2 * q0w * s001 * theta - 2 * q0z * s011 * theta),
                q0w * qperpy * s002 - q0x * qperpz * s002 - q0x * qperpw * s012 - q0w * qperpx * s012 +
                    2 * q0x * qperpx * s022 - q0w * qperpy * s102 + q0x * qperpz * s102 + q0x * qperpw * s112 +
                    q0w * qperpx * s112 - 2 * q0x * qperpx * s122 - 2 * qperpw * qperpy * s002 * theta +
                    2 * qperpx * qperpz * s002 * theta - 2 * q0w * q0x * s012 * theta + 2 * qperpw * qperpx * s012 * theta +
                    2 * qperpy * qperpz * s012 * theta + 2 * q0x * q0x * s022 * theta + 2 * q0y * q0y * s022 * theta -
                    2 * qperpx * qperpx * s022 * theta - 2 * qperpy * qperpy * s022 * theta +
                    q0z * (-(qperpx * s002) - qperpy * s012 + qperpx * s102 + qperpy * s112 - 2 * q0x * s002 * theta) +
                    q0y * (-(qperpz * s012) + 2 * qperpy * s022 + qperpw * (s002 - s102) + qperpz * s112 -
                           2 * qperpy * s122 + 2 * q0w * s002 * theta - 2 * q0z * s012 * theta));

            m_c5[2] = DerivativeTerm(0.,
                                     2 *
                                         (qperpw * qperpy * s000 - qperpx * qperpz * s000 + q0y * q0z * s010 -
                                          qperpw * qperpx * s010 - qperpy * qperpz * s010 - q0y * q0y * s020 +
                                          qperpx * qperpx * s020 + qperpy * qperpy * s020 + q0x * q0z * (s000 - s100) -
                                          qperpw * qperpy * s100 + qperpx * qperpz * s100 +
                                          q0w * (q0y * (-s000 + s100) + q0x * (s010 - s110)) - q0y * q0z * s110 +
                                          qperpw * qperpx * s110 + qperpy * qperpz * s110 + q0y * q0y * s120 -
                                          qperpx * qperpx * s120 - qperpy * qperpy * s120 + q0x * q0x * (-s020 + s120)) *
                                         theta,
                                     2 *
                                         (qperpw * qperpy * s001 - qperpx * qperpz * s001 + q0y * q0z * s011 -
                                          qperpw * qperpx * s011 - qperpy * qperpz * s011 - q0y * q0y * s021 +
                                          qperpx * qperpx * s021 + qperpy * qperpy * s021 + q0x * q0z * (s001 - s101) -
                                          qperpw * qperpy * s101 + qperpx * qperpz * s101 +
                                          q0w * (q0y * (-s001 + s101) + q0x * (s011 - s111)) - q0y * q0z * s111 +
                                          qperpw * qperpx * s111 + qperpy * qperpz * s111 + q0y * q0y * s121 -
                                          qperpx * qperpx * s121 - qperpy * qperpy * s121 + q0x * q0x * (-s021 + s121)) *
                                         theta,
                                     2 *
                                         (qperpw * qperpy * s002 - qperpx * qperpz * s002 + q0y * q0z * s012 -
                                          qperpw * qperpx * s012 - qperpy * qperpz * s012 - q0y * q0y * s022 +
                                          qperpx * qperpx * s022 + qperpy * qperpy * s022 + q0x * q0z * (s002 - s102) -
                                          qperpw * qperpy * s102 + qperpx * qperpz * s102 +
                                          q0w * (q0y * (-s002 + s102) + q0x * (s012 - s112)) - q0y * q0z * s112 +
                                          qperpw * qperpx * s112 + qperpy * qperpz * s112 + q0y * q0y * s122 -
                                          qperpx * qperpx * s122 - qperpy * qperpy * s122 + q0x * q0x * (-s022 + s122)) *
                                         theta);
        }
    }

    __host__ __device__ bool AnimatedTransform::isAnimated() const { return (m_state & eAnimated) != 0; }

    __host__ __device__ Vector3f AnimatedTransform::applyInverse(Vector3f vpn, float time) const
    {
        Vector3f ret;
        if (!isAnimated() || time <= startTime)
            ret = startTransform.applyInverse(vpn);
        else if (time >= endTime)
            ret = endTransform.applyInverse(vpn);
        else
            ret = interpolate(time).applyInverse(vpn);

        return ret;
    }
    __host__ __device__ Point3f AnimatedTransform::applyInverse(Point3f vpn, float time) const
    {
        Point3f ret;
        if (!isAnimated() || time <= startTime)
            ret = startTransform.applyInverse(vpn);
        else if (time >= endTime)
            ret = endTransform.applyInverse(vpn);
        else
            ret = interpolate(time).applyInverse(vpn);

        return ret;
    }
    __host__ __device__ Normal3f AnimatedTransform::applyInverse(Normal3f vpn, float time) const
    {
        Normal3f ret;
        if (!isAnimated() || time <= startTime)
            ret = startTransform.applyInverse(vpn);
        else if (time >= endTime)
            ret = endTransform.applyInverse(vpn);
        else
            ret = interpolate(time).applyInverse(vpn);

        return ret;
    }
    __host__ __device__ Vector3f AnimatedTransform::operator()(Vector3f vpn, float time) const
    {
        Vector3f ret;
        if (!isAnimated() || time <= startTime)
            ret = startTransform(vpn);
        else if (time >= endTime)
            ret = endTransform(vpn);
        else
        {
            Transform t = interpolate(time);
            ret         = t(vpn);
        }

        return ret;
    }
    __host__ __device__ Point3f AnimatedTransform::operator()(Point3f vpn, float time) const
    {
        Point3f ret;
        if (!isAnimated() || time <= startTime)
            ret = startTransform(vpn);
        else if (time >= endTime)
            ret = endTransform(vpn);
        else
        {
            Transform t = interpolate(time);
            ret         = t(vpn);
        }

        return ret;
    }
    __host__ __device__ Normal3f AnimatedTransform::operator()(Normal3f vpn, float time) const
    {
        Normal3f ret;
        if (!isAnimated() || time <= startTime)
            ret = startTransform(vpn);
        else if (time >= endTime)
            ret = endTransform(vpn);
        else
        {
            Transform t = interpolate(time);
            ret         = t(vpn);
        }

        return ret;
    }

    __host__ __device__ Ray AnimatedTransform::operator()(Ray const& ray, float* optInOut_tMax) const
    {
        Ray ret = ray;
        if (!isAnimated() || ray.time <= startTime)
            ret = startTransform(ray, optInOut_tMax);
        else if (ray.time >= endTime)
            ret = endTransform(ray, optInOut_tMax);
        else
        {
            Transform t = interpolate(ray.time);
            ret         = t(ray, optInOut_tMax);
        }
        return ret;
    }
    __host__ __device__ RayDifferential AnimatedTransform::operator()(RayDifferential const& ray, float* optInOut_tMax) const
    {
        RayDifferential ret = ray;
        if (!isAnimated() || ray.time <= startTime)
            ret = startTransform(ray, optInOut_tMax);
        else if (ray.time >= endTime)
            ret = endTransform(ray, optInOut_tMax);
        else
        {
            Transform t = interpolate(ray.time);
            ret         = t(ray, optInOut_tMax);
        }
        return ret;
    }

    __host__ __device__ Ray AnimatedTransform::applyInverse(Ray const& ray, float* optInOut_tMax) const
    {
        Ray ret = ray;
        if (!isAnimated() || ray.time <= startTime)
            ret = startTransform.applyInverse(ray, optInOut_tMax);
        else if (ray.time >= endTime)
            ret = endTransform.applyInverse(ray, optInOut_tMax);
        else
        {
            Transform t = interpolate(ray.time);
            ret         = t.applyInverse(ray, optInOut_tMax);
        }
        return ret;
    }

    __host__ __device__ RayDifferential AnimatedTransform::applyInverse(RayDifferential const& ray, float* optInOut_tMax) const
    {
        RayDifferential ret = ray;
        if (!isAnimated() || ray.time <= startTime)
            ret = startTransform.applyInverse(ray, optInOut_tMax);
        else if (ray.time >= endTime)
            ret = endTransform.applyInverse(ray, optInOut_tMax);
        else
        {
            Transform t = interpolate(ray.time);
            ret         = t.applyInverse(ray, optInOut_tMax);
        }
        return ret;
    }

    __host__ __device__ bool AnimatedTransform::hasScale() const
    {
        return startTransform.hasScale() || endTransform.hasScale();
    }

    __host__ __device__ bool AnimatedTransform::hasRotation() const { return (m_state & eHasRotation) != 0; }

    __host__ __device__ Transform AnimatedTransform::interpolate(float time) const
    {
        // Handle boundary conditions for matrix interpolation
        if (!isAnimated() || time <= startTime)
            return startTransform;
        if (time >= endTime)
            return endTransform;

        float dt = (time - startTime) / (endTime - startTime);

        // Interpolate translation at _dt_
        Vector3f trans = (1 - dt) * m_t[0] + dt * m_t[1];

        // Interpolate rotation at _dt_
        Quaternion rotate = slerp(dt, m_r[0], m_r[1]);

        // Interpolate scale at _dt_
        Matrix4f scale = (1 - dt) * m_s[0] + dt * m_s[1];

        // Construct the final transformation matrix
        Matrix4f transform = Matrix4f::identity(); // Start with identity

        // Apply scale (scale matrix)
        transform = scale;

        // Apply rotation (rotation matrix)
        Matrix4f rotationMatrix = fromQuat(rotate); // Convert quaternion to rotation matrix
        transform               = rotationMatrix * transform;

        // Apply translation (translation matrix)
        transform[3] = {{trans.x, trans.y, trans.z, 1.0f}}; // Set the translation column

        return Transform(transform); // Assuming Transform wraps glm::mat4
    }

    __host__ __device__ Bounds3f AnimatedTransform::motionBounds(Bounds3f const& b) const
    {
        Bounds3f bounds;
        // Handle easy cases for _Bounds3f_ motion bounds
        if (!isAnimated())
            bounds = startTransform(b);
        if (!hasRotation())
            bounds = bbUnion(startTransform(b), endTransform(b));
        else // Return motion bounds accounting for animated rotation
        {
            for (int32_t corner = 0; corner < 8; ++corner)
                bounds = bbUnion(bounds, boundsPointMotion(b.corner(static_cast<EBoundsCorner>(corner))));
        }

        return bounds;
    }

    __host__ __device__ Bounds3f AnimatedTransform::boundsPointMotion(Point3f p) const
    {
        if (!isAnimated())
        {
            Point3f thepoint = startTransform(p);
            return {thepoint, thepoint};
        }

        Bounds3f bounds{startTransform(p), endTransform(p)};
        float    cosTheta = dot(m_r[0], m_r[1]);
        float    theta    = glm::acos(glm::clamp(cosTheta, -1.f, 1.f));
        for (int c = 0; c < 3; ++c)
        {
            // Find any motion derivative zeros for the component _c_
            float zeros[8];
            int   nZeros = 0;
            findZeros(m_c1[c].eval(p),
                      m_c2[c].eval(p),
                      m_c3[c].eval(p),
                      m_c4[c].eval(p),
                      m_c5[c].eval(p),
                      theta,
                      Intervalf(0.f, 1.f),
                      {zeros, 8},
                      nZeros);

            // Expand bounding box for any motion derivative zeros found
            for (int i = 0; i < nZeros; ++i)
            {
                Point3f pz = (*this)(p, glm::lerp(startTime, endTime, zeros[i]));
                bounds     = bbUnion(bounds, pz);
            }
        }
        return bounds;
    }

    __host__ __device__ float AnimatedTransform::DerivativeTerm::eval(Point3f p) const
    {
        return kc + kx * p.x + ky * p.y + kz * p.z;
    }

    __host__ __device__ void AnimatedTransform::findZeros(
        float            c1,
        float            c2,
        float            c3,
        float            c4,
        float            c5,
        float            theta,
        Intervalf        tInterval,
        ArrayView<float> zeros,
        int32_t&         outNumZeros,
        int32_t          depth)
    {
        static constexpr float tolerance = 1e-3f;
        assert(zeros.length >= 8);
        // binary search + refinement through newton's methods = we need the derivative estimate to guide a first order gradient descent
        // motion derivative in interval form
        Intervalf const cosWeight = Intervalf(c2) + Intervalf(c3) * tInterval;
        Intervalf const sinWeight = Intervalf(c4) + Intervalf(c5) * tInterval;
        Intervalf const angle     = Intervalf(2 * theta) * tInterval;
        Intervalf const dadt      = Intervalf(c1)                                                   //
                               + (cosWeight * Intervalf{glm::cos(angle.low), glm::cos(angle.high)}) //
                               + (sinWeight * Intervalf{glm::sin(angle.low), glm::sin(angle.high)});
        // if there is no accceleration, return
        if (dadt.low > 0 || dadt.high < 0 || glm::epsilonEqual(dadt.low, dadt.high, fl::eqTol()))
            return;

        if (depth > 0 && dadt.width() > tolerance)
        { // zero not found in interval, recurse
            float mid = tInterval.midpoint();
            findZeros(c1, c2, c3, c4, c5, theta, Intervalf(tInterval.low, mid), zeros, outNumZeros, depth - 1);
            findZeros(c1, c2, c3, c4, c5, theta, Intervalf(mid, tInterval.high), zeros, outNumZeros, depth - 1);
        }
        else
        { // zero found or max depth reached, refine (4 iterations)
            float tNewton = tInterval.midpoint();
            for (int32_t i = 0; i < 4; ++i)
            {
                float fNewton = c1 + (c2 + c3 * tNewton) * glm::cos(2 * theta * tNewton) +
                                (c4 + c5 * tNewton) * glm::sin(2 * theta * tNewton);
                float fPrimeNewton =                                                         //
                    (c3 + 2 * (c4 + c5 * tNewton) * theta) * glm::cos(2 * tNewton * theta) + //
                    (c5 - 2 * (c2 + c3 * tNewton) * theta) * glm::sin(2 * tNewton * theta);  //
                if (glm::epsilonEqual(fNewton, 0.f, fl::eqTol()) || glm::epsilonEqual(fPrimeNewton, 0.f, fl::eqTol()))
                    break;
                tNewton -= fNewton / fPrimeNewton;
            }

            // record zero if it is in the time interval
            if (tNewton >= tInterval.low - tolerance && tNewton < tInterval.high + tolerance)
            {
                assert(zeros.length < outNumZeros);
                zeros.data[outNumZeros++] = tNewton;
            }
        }
    }
#endif

} // namespace dmt
