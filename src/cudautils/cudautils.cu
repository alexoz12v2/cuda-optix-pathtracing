#include "cudautils.h"

#include <platform/platform.h>

// silence warnings __host__ __device__ on a defaulted copy control
#if defined(__NVCC__)
#pragma nv_diag_suppress 20012
#endif
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp>  // glm::pi
#include <glm/geometric.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/matrix_decompose.hpp> // glm::decompose
#include <glm/gtx/norm.hpp>             // glm::length2
#if defined(__NVCC__)
#pragma nv_diag_default 20012
#endif

namespace dmt {
    // math utilities: float ------------------------------------------------------------------------------------------
    __host__ __device__ float Intervalf::midpoint() const { return (low + high) / 2; }

    __host__ __device__ float Intervalf::width() const { return high - low; }

    __host__ __device__ Intervalf operator+(Intervalf a, Intervalf b)
    {
        return {fl::addRoundDown(a.low, b.low), fl::addRoundUp(a.high, b.high)};
    }

    __host__ __device__ Intervalf operator-(Intervalf a, Intervalf b)
    {
        return {fl::subRoundDown(a.low, b.low), fl::subRoundUp(a.high, b.high)};
    }

    __host__ __device__ Intervalf operator*(Intervalf a, Intervalf b)
    {
        float lp[4] = {fl::mulRoundDown(a.low, b.low),
                       fl::mulRoundDown(a.high, b.low),
                       fl::mulRoundDown(a.low, b.high),
                       fl::mulRoundDown(a.high, b.high)};
        float hp[4] = {fl::mulRoundUp(a.low, b.low),
                       fl::mulRoundUp(a.high, b.low),
                       fl::mulRoundUp(a.low, b.high),
                       fl::mulRoundUp(a.high, b.high)};
        return {std::min({lp[0], lp[1], lp[2], lp[3]}), std::max({hp[0], hp[1], hp[2], hp[3]})};
    }

    __host__ __device__ Intervalf operator/(Intervalf a, Intervalf b)
    {
        // if the interval of the divisor contains zero, return the whole extended real number line
        if (b.low < 0.f && b.high > 0.f)
            return {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};

        float lowQuot[4]  = {fl::divRoundDown(a.low, b.low),
                             fl::divRoundDown(a.high, b.low),
                             fl::divRoundDown(a.low, b.high),
                             fl::divRoundDown(a.high, b.high)};
        float highQuot[4] = {fl::divRoundUp(a.low, b.low),
                             fl::divRoundUp(a.high, b.low),
                             fl::divRoundUp(a.low, b.high),
                             fl::divRoundUp(a.high, b.high)};
        return {std::min({lowQuot[0], lowQuot[1], lowQuot[2], lowQuot[3]}),
                std::max({highQuot[0], highQuot[1], highQuot[2], highQuot[3]})};
    }

    // math utilities: vector -----------------------------------------------------------------------------------------
    __host__ __device__ Transform::Transform() : m(Mat4f(1.0f)), mInv(Mat4f(1.0f)) {}
    __host__ __device__ Transform::Transform(Mat4f const& matrix) : m(matrix), mInv(glm::inverse(matrix)) {}

    __host__ __device__ void Transform::translate_(Vec3f const& translation)
    {
        m    = glm::translate(m, translation);
        mInv = glm::translate(mInv, -translation);
    }

    // Apply scaling
    __host__ __device__ void Transform::scale_(Vec3f const& scaling)
    {
        m    = glm::scale(m, scaling);
        mInv = glm::scale(mInv, 1.0f / scaling);
    }

    // Apply rotation (angle in degrees)
    __host__ __device__ void Transform::rotate_(float angle, Vec3f const& axis)
    {
        m    = glm::rotate(m, glm::radians(angle), axis);
        mInv = glm::rotate(mInv, -glm::radians(angle), axis);
    }

    // Combine with another transform
    __host__ __device__ Transform Transform::combine(Transform const& other) const
    {
        Transform result;
        result.m    = m * other.m;
        result.mInv = other.mInv * mInv;
        return result;
    }

    // Combine with another transform
    __host__ __device__ void Transform::combine_(Transform const& other)
    {
        m    = m * other.m;
        mInv = other.mInv * mInv;
    }

    __host__ __device__ void Transform::lookAt_(Vec3f pos, Vec3f look, Vec3f up)
    {
        Mat4f worldFromCamera;
        // Initialize fourth column of viewing matrix
        worldFromCamera[0][3] = pos.x;
        worldFromCamera[1][3] = pos.y;
        worldFromCamera[2][3] = pos.z;
        worldFromCamera[3][3] = 1;

        // Initialize first three columns of viewing matrix
        Vec3f dir = glm::normalize(look - pos);
        assert(glm::length(glm::cross(glm::normalize(up), dir)) < std::numeric_limits<float>::epsilon());

        Vec3f right           = glm::normalize(glm::cross(glm::normalize(up), dir));
        Vec3f newUp           = glm::cross(dir, right);
        worldFromCamera[0][0] = right.x;
        worldFromCamera[1][0] = right.y;
        worldFromCamera[2][0] = right.z;
        worldFromCamera[3][0] = 0.;
        worldFromCamera[0][1] = newUp.x;
        worldFromCamera[1][1] = newUp.y;
        worldFromCamera[2][1] = newUp.z;
        worldFromCamera[3][1] = 0.;
        worldFromCamera[0][2] = dir.x;
        worldFromCamera[1][2] = dir.y;
        worldFromCamera[2][2] = dir.z;
        worldFromCamera[3][2] = 0.;

        m    = m * worldFromCamera;
        mInv = glm::inverse(worldFromCamera) * mInv;
    }

    __host__ __device__ void Transform::concatTrasform_(std::array<float, 16> const& t)
    {
        Mat4f mt{t[0], t[1], t[2], t[3], /**/ t[4], t[5], t[6], t[7], /**/ t[8], t[9], t[10], t[11], /**/ t[12], t[13], t[14], t[15]};
        Transform concatT{glm::transpose(m)};
        m    = m * concatT.m;
        mInv = concatT.mInv * mInv;
    }

    __host__ __device__ void Transform::reset_()
    {
        m    = Mat4f(1.0f);
        mInv = Mat4f(1.0f);
    }


    __host__ __device__ void Transform::inverse_()
    {
        Mat4f tmp = m;
        m         = mInv;
        mInv      = tmp;
    }


    __host__ __device__ Vec3f Transform::applyToPoint(Vec3f const& point) const
    {
        Vec4f result = m * Vec4f(point, 1.0f);
        return Vec3f(result);
    }

    // Apply the inverse transform to a point
    __host__ __device__ Vec3f Transform::applyInverseToPoint(Vec3f const& point) const
    {
        Vec4f result = mInv * Vec4f(point, 1.0f);
        return Vec3f(result);
    }

    __host__ __device__ void Transform::decompose(Vec3f& outT, Quat& outR, Mat4f& outS) const
    {
        // discarded components
        Vec3f scale;
        Vec3f skew;
        Vec4f perspective;
        glm::decompose(m, scale, outR, outT, skew, perspective);
        // decompose actually returs the conjugate quaternion
        outR = glm::conjugate(outR);
        // inglobe all the rest into a matrixc
        // Start with an identity matrix
        outS = Mat4f(1.f);

        // Apply scaling
        outS[0][0] = scale.x;
        outS[1][1] = scale.y;
        outS[2][2] = scale.z;

        // Apply skew (off-diagonal elements)
        outS[1][0] = skew.x; // Skew Y by X
        outS[2][0] = skew.y; // Skew Z by X
        outS[2][1] = skew.z; // Skew Z by Y

        // Apply perspective (set the last row)
        outS[3] = perspective;
    }


    // Equality comparison
    __host__ __device__ bool Transform::operator==(Transform const& other) const
    {
        return m == other.m && mInv == other.mInv;
    }

    // Inequality comparison
    __host__ __device__ bool Transform::operator!=(Transform const& other) const { return !(*this == other); }

    __host__ __device__ bool Transform::hasScale(float tolerance) const
    {
        // compute the length of the three reference unit vectors after being transformed. if any of these has been
        // scaled, then the transformation has a scaling component
        using enum EVecType32;
        Vec3f const scales{glm::length2(operator()(Vec3f{1.f, 0.f, 0.f}, eVector)),
                           glm::length2(operator()(Vec3f{0.f, 1.f, 0.f}, eVector)),
                           glm::length2(operator()(Vec3f{0.f, 0.f, 1.f}, eVector))};
        auto        bVec = glm::epsilonEqual(scales, Vec3f{1.f, 1.f, 1.f}, tolerance);
        return !(bVec.x && bVec.y && bVec.z);
    }

    __host__ __device__ Vec3f Transform::applyInverse(Vec3f vpn, EVecType32 type, bool normalize) const
    {
        Vec3f     ret; // NRVO
        glm::vec4 temp{vpn.x, vpn.y, vpn.z, fl::bitsToFloat(type)};
        temp = mInv * temp;
        temp /= temp.w;
        ret.x = temp.x;
        ret.y = temp.y;
        ret.z = temp.z;
        if (normalize)
        {
            assert(type == EVecType32::eVector);
            ret = glm::normalize(ret);
        }

        return ret;
    }

    __host__ __device__ Vec3f Transform::operator()(Vec3f vpn, EVecType32 type, bool normalize) const
    {
        Vec3f     ret; // NRVO
        glm::vec4 temp{vpn.x, vpn.y, vpn.z, fl::bitsToFloat(type)};
        temp = m * temp;
        temp /= temp.w;
        ret.x = temp.x;
        ret.y = temp.y;
        ret.z = temp.z;
        if (normalize)
        {
            assert(type == EVecType32::eVector);
            ret = glm::normalize(ret);
        }

        return ret;
    }

    __host__ __device__ Bounds3f Transform::operator()(Bounds3f const& b) const
    {
        using enum EVecType32;
        Bounds3f bRet;
        for (int i = 0; i < 8 /*corners*/; ++i)
        {
            Pt3f point = b.corner(static_cast<Bounds3f::EBoundsCorner>(i));
            bRet       = bbUnion(bRet, (*this)(point, ePoint));
        }
        return bRet;
    }

    // Bounds3f -------------------------------------------------------------------------------------------------------
    __host__ __device__ bool inside(Pt3f p, Bounds3f const& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y && p.y <= b.pMax.y && p.z >= b.pMin.z &&
                p.z <= b.pMax.z);
    }

    __host__ __device__ bool almostEqual(Mat4f const& a, Mat4f const& b)
    {
        bool result = true;
        for (uint32_t i = 0; i < 4; ++i)
        {
            auto bVec = glm::epsilonEqual(a[i], b[i], fl::eqTol());
            result    = result && bVec.x && bVec.y && bVec.z;
            if (!result)
                return result;
        }
        return result;
    }

    __host__ __device__ Bounds3f bbUnion(Bounds3f const& a, Bounds3f const& b)
    {
        Bounds3f bRet;
        bRet.pMin = glm::min(a.pMin, b.pMin);
        bRet.pMax = glm::max(a.pMax, b.pMax);
        return bRet;
    }

    __host__ __device__ Bounds3f bbUnion(Bounds3f const& b, Pt3f p)
    {
        Bounds3f bRet;
        bRet.pMin = glm::min(b.pMin, p);
        bRet.pMax = glm::max(b.pMax, p);
        return bRet;
    }

    __host__ __device__ Vec3f& Bounds3f::operator[](int32_t i)
    {
        assert(i == 0 || i == 1);
        return i == 0 ? pMin : pMax;
    }

    __host__ __device__ Vec3f const& Bounds3f::operator[](int32_t i) const
    {
        assert(i == 0 || i == 1);
        return i == 0 ? pMin : pMax;
    }

    __host__ __device__ Pt3f Bounds3f::corner(EBoundsCorner corner) const
    {
        Pt3f const ret{
            operator[](corner& eRight).x,
            operator[]((corner & eForward) >> 1).y,
            operator[]((corner & eTop) >> 2).z,
        };
        return ret;
    }

    __host__ __device__ Vec3f Bounds3f::diagonal() const { return pMax - pMin; }

    __host__ __device__ float Bounds3f::surfaceAraa() const
    {
        Vec3f const d = diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    __host__ __device__ float Bounds3f::volume() const
    {
        Vec3f const d = diagonal();
        return d.x * d.y * d.z;
    }

    __host__ __device__ int32_t Bounds3f::maxDimention() const
    {
        Vec3f const d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    __host__ __device__ Pt3f Bounds3f::lerp(Pt3f t) const
    {
        Pt3f const ret{glm::lerp(pMin, pMax, t)};
        return ret;
    }

    __host__ __device__ Vec3f Bounds3f::offset(Pt3f p) const
    {
        Vec3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    __host__ __device__ void Bounds3f::boundingSphere(Pt3f& outCenter, float& outRadius) const
    {
        outCenter = (pMin + pMax) / 2.f;
        outRadius = inside(outCenter, *this) ? glm::distance(outCenter, pMax) : 0.f;
    }

    __host__ __device__ bool Bounds3f::isEmpty() const
    {
        return pMin.x >= pMax.x || pMin.y >= pMax.y || pMin.z >= pMax.z;
    }

    __host__ __device__ bool Bounds3f::isDegenerate() const
    {
        return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
    }

    __host__ __device__ bool Bounds3f::operator==(Bounds3f const& that) const
    {
        auto bVec = glm::epsilonEqual(pMin, that.pMin, fl::eqTol()) && glm::epsilonEqual(pMax, that.pMax, fl::eqTol());
        return bVec.x && bVec.y && bVec.z;
    }

    __host__ __device__ bool Bounds3f::intersectP(Pt3f o, Vec3f d, float tMax, float* DMT_RESTRICT hit0, float* DMT_RESTRICT hit1) const
    {
        float t0 = 0, t1 = tMax;
        for (int32_t i = 0; i < 3; ++i)
        {
            float invRayDir = 1 / d[i];
            float tNear     = (pMin[i] - o[i]) * invRayDir;
            float tFar      = (pMax[i] - o[i]) * invRayDir;
            if (tNear > tFar)
            {
                float tmp = tNear;
                tNear     = tFar;
                tFar      = tmp;
            }
            tFar *= 1 + 2 * fl::gamma(3);

            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if (t0 > t1)
                return false;
        }
        if (hit0)
            *hit0 = t0;
        if (hit1)
            *hit1 = t1;
        return true;
    }

    __host__ __device__ bool Bounds3f::intersectP(Pt3f o, Vec3f d, float rayMax, Vec3f invDir, std::array<int32_t, 3> dirIsNeg) const
    {
        Bounds3f const& self = *this;
        // check ray intersection agains x and y slabs
        float tMin  = (self[dirIsNeg[0]].x - o.x) * invDir.x;
        float tMax  = (self[1 - dirIsNeg[0]].x - o.x) * invDir.x;
        float tyMin = (self[dirIsNeg[1]].y - o.y) * invDir.y;
        float tyMax = (self[1 - dirIsNeg[1]].y - o.y) * invDir.y;

        // update tMax and tyMax to ensure robust intersection
        tMax *= 1 + 2 * fl::gamma(3);
        tyMax *= 1 + 2 * fl::gamma(3);

        if (tMin > tyMax || tyMin > tMax)
            return false;
        if (tyMin > tMin)
            tMin = tyMin;
        if (tyMax < tMax)
            tMax = tyMax;

        // check ray intersection against z slab
        float tzMin = (self[dirIsNeg[2]].z - o.z) * invDir.z;
        float tzMax = (self[1 - dirIsNeg[2]].z - o.z) * invDir.z;
        tzMax *= 1 + 2 * fl::gamma(3);

        if (tMin > tzMax || tzMin > tMax)
            return false;
        if (tzMin > tMin)
            tMin = tzMin;
        if (tzMax < tMax)
            tMax = tzMax;

        return (tMin < rayMax) && (tMax > 0);
    }


    AnimatedTransform::DerivativeTerm::DerivativeTerm() : kc(0.f), kx(0.f), ky(0.f), kz(0.f) {}

    // AnimatedTransform ----------------------------------------------------------------------------------------------
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
    m_state(almostEqual(startTransform.m, endTransform.m) ? eAnimated : eNone)
    {
        if ((m_state & eAnimated) == 0)
            return;
        // Decompose start and end transformations
        startTransform.decompose(m_t[0], m_r[0], m_s[0]);
        endTransform.decompose(m_t[1], m_r[1], m_s[1]);

        // Flip _R[1]_ if needed to select shortest path
        if (glm::dot(m_r[0], m_r[1]) < 0)
            m_r[1] = -m_r[1];

        if (glm::dot(m_r[0], m_r[1]) < 0.9995f)
            m_state = static_cast<EState>(m_state | eHasRotation);
        // Compute terms of motion derivative function
        if ((m_state & eHasRotation) != 0)
        {
            float cosTheta = glm::dot(m_r[0], m_r[1]);
            float theta    = glm::acos(glm::clamp(cosTheta, -1.f, 1.f));
            Quat  qperp    = glm::normalize(m_r[1] - m_r[0] * cosTheta);

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

    __host__ __device__ Vec3f AnimatedTransform::applyInverse(Vec3f vpn, float time, EVecType32 type, bool normalize) const
    {
        Vec3f ret; // NRVO
        if (!isAnimated())
            ret = startTransform.applyInverse(vpn, type, normalize);
        else
            ret = interpolate(time).applyInverse(vpn, type, normalize);

        return ret;
    }

    __host__ __device__ Vec3f AnimatedTransform::operator()(Vec3f vpn, float time, EVecType32 type, bool normalize) const
    {
        Vec3f ret; // NRVO
        if (!isAnimated() || time <= startTime)
            ret = startTransform(vpn, type, normalize);
        else if (time >= endTime)
            ret = endTransform(vpn, type, normalize);
        else
        {
            Transform t = interpolate(time);
            ret         = t(vpn, type, normalize);
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
        Vec3f trans = (1 - dt) * m_t[0] + dt * m_t[1]; // typedef glm::vec3

        // Interpolate rotation at _dt_
        Quat rotate = glm::slerp(m_r[0], m_r[1], dt); // glm::quat

        // Interpolate scale at _dt_
        Mat4f scale = (1 - dt) * m_s[0] + dt * m_s[1]; // glm::mat4

        // Construct the final transformation matrix
        Mat4f transform = glm::mat4(1.0f); // Start with identity

        // Apply scale (scale matrix)
        transform = scale;

        // Apply rotation (rotation matrix)
        Mat4f rotationMatrix = glm::mat4_cast(rotate); // Convert quaternion to rotation matrix
        transform            = rotationMatrix * transform;

        // Apply translation (translation matrix)
        transform[3] = glm::vec4(trans, 1.0f); // Set the translation column

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
                bounds = bbUnion(bounds, boundsPointMotion(b.corner(static_cast<Bounds3f::EBoundsCorner>(corner))));
        }

        return bounds;
    }

    __host__ __device__ Bounds3f AnimatedTransform::boundsPointMotion(Pt3f p) const
    {
        using enum EVecType32;
        if (!isAnimated())
            return Bounds3f(startTransform(p, ePoint));

        Bounds3f bounds(startTransform(p, ePoint), endTransform(p, ePoint));
        float    cosTheta = glm::dot(m_r[0], m_r[1]);
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
                Pt3f pz = (*this)(p, glm::lerp(startTime, endTime, zeros[i]), ePoint);
                bounds  = bbUnion(bounds, pz);
            }
        }
        return bounds;
    }

    __host__ __device__ float AnimatedTransform::DerivativeTerm::eval(Pt3f p) const
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

    // CameraTransform ------------------------------------------------------------------------------------------------
    __host__ __device__ CameraTransform::CameraTransform(AnimatedTransform const& worldFromCamera, ERenderCoordSys renderCoordSys)
    {
#if !defined(__CUDA_ARCH__)
        // TODO wrap AppContextJanitor in another janitor class to handle cuda execution
        AppContextJanitor j;
#endif
        switch (renderCoordSys)
        {
            using enum ERenderCoordSys;
            case eCameraWorld:
            {

                break;
            }
            case eCamera:
            {
                break;
            }
            case eWorld:
            {
                break;
            }
            default:
#if !defined(__CUDA_ARCH__)
                j.actx.error("Unexpected render coordinate system, aborting");
                std::abort();
#else
                // TODO: per-warp buffers inside managed memory watched by a job, and logged when activated
                __threadfence();
                asm("trap;");
#endif
                break;
        }
    }
} // namespace dmt

namespace dmt::soa {
    using namespace dmt;
}