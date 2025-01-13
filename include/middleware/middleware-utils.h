#pragma once

#include "dmtmacros.h"
#include "middleware/middleware-macros.h"

#if !defined(DMT_NEEDS_MODULE)
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp>  // glm::pi
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/vec3.hpp>   // glm::vec3
#include <glm/vec4.hpp>   // glm::vec4
// includes duplicated from middleware.cppm
#endif

DMT_MODULE_EXPORT namespace dmt {
    DMT_MIDDLEWARE_API       glm::mat4;
    class DMT_MIDDLEWARE_API Transform
    {
    public:
        glm::mat4 m;    // Transformation matrix
        glm::mat4 mInv; // Inverse transformation matrix

        // Default constructor
        Transform();

        // Constructor with an initial matrix
        explicit Transform(glm::mat4 const& matrix);

        // Apply translation
        void translate_(glm::vec3 const& translation);

        // Apply scaling
        void scale_(glm::vec3 const& scaling);

        // Apply rotation (angle in degrees)
        void rotate_(float angle, glm::vec3 const& axis);

        // Combine with another transform
        Transform combine(Transform const& other) const;

        // Combine with another transform
        void combine_(Transform const& other);

        void lookAt_(glm::vec3 pos, glm::vec3 look, glm::vec3 up);

        // Reset to identity matrix
        void reset();

        // Swap m and mInv
        void inverse();

        // Apply the transform to a point
        glm::vec3 applyToPoint(glm::vec3 const& point) const;

        // Apply the inverse transform to a point
        glm::vec3 applyInverseToPoint(glm::vec3 const& point) const;

        // Equality comparison
        bool operator==(Transform const& other) const;

        // Inequality comparison
        bool operator!=(Transform const& other) const;
    };
}