#define DMT_INTERFACE_AS_HEADER
#undef DMT_NEEDS_MODULE
#include "middleware-utils.h"

namespace dmt {
    Transform::Transform() : m(glm::mat4(1.0f)), mInv(glm::mat4(1.0f)) {}
    Transform::Transform(glm::mat4 const& matrix) : m(matrix), mInv(glm::inverse(matrix)) {}

    void Transform::translate_(glm::vec3 const& translation)
    {
        m    = glm::translate(m, translation);
        mInv = glm::translate(mInv, -translation);
    }

    // Apply scaling
    void Transform::scale_(glm::vec3 const& scaling)
    {
        m    = glm::scale(m, scaling);
        mInv = glm::scale(mInv, 1.0f / scaling);
    }

    // Apply rotation (angle in degrees)
    void Transform::rotate_(float angle, glm::vec3 const& axis)
    {
        m    = glm::rotate(m, glm::radians(angle), axis);
        mInv = glm::rotate(mInv, -glm::radians(angle), axis);
    }

    // Combine with another transform
    Transform Transform::combine(Transform const& other) const
    {
        Transform result;
        result.m    = m * other.m;
        result.mInv = other.mInv * mInv;
        return result;
    }

    // Combine with another transform
    void Transform::combine_(Transform const& other)
    {
        m    = m * other.m;
        mInv = other.mInv * mInv;
    }

    void Transform::lookAt_(glm::vec3 pos, glm::vec3 look, glm::vec3 up)
    {
        glm::mat4 worldFromCamera;
        // Initialize fourth column of viewing matrix
        worldFromCamera[0][3] = pos.x;
        worldFromCamera[1][3] = pos.y;
        worldFromCamera[2][3] = pos.z;
        worldFromCamera[3][3] = 1;

        // Initialize first three columns of viewing matrix
        glm::vec3 dir = glm::normalize(look - pos);
        assert(glm::length(glm::cross(glm::normalize(up), dir)) < std::numeric_limits<float>::epsilon());

        glm::vec3 right       = glm::normalize(glm::cross(glm::normalize(up), dir));
        glm::vec3 newUp       = glm::cross(dir, right);
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

        m = m * worldFromCamera;
        mInv = glm::inverse(worldFromCamera) * mInv;
    }

    void concatTrasform_(std::array<float, 16> const& transform)
    {
        Transform concatT = glm::traspose(glm::mat4(transform.data()));
        m                 = m * concatT;
        mInv              = concatT.mInv * mInv;
    }

    // Reset to identity matrix
    void Transform::reset()
    {
        m    = glm::mat4(1.0f);
        mInv = glm::mat4(1.0f);
    }

    // Swap m and mInv
    void Transform::inverse()
    {
        glm::mat4 tmp = m;
        m             = mInv;
        mInv          = tmp;
    }


    // Apply the transform to a point
    glm::vec3 Transform::applyToPoint(glm::vec3 const& point) const
    {
        glm::vec4 result = m * glm::vec4(point, 1.0f);
        return glm::vec3(result);
    }

    // Apply the inverse transform to a point
    glm::vec3 Transform::applyInverseToPoint(glm::vec3 const& point) const
    {
        glm::vec4 result = mInv * glm::vec4(point, 1.0f);
        return glm::vec3(result);
    }


    // Equality comparison
    bool Transform::operator==(Transform const& other) const { return m == other.m && mInv == other.mInv; }

    // Inequality comparison
    bool Transform::operator!=(Transform const& other) const { return !(*this == other); }
} // namespace dmt