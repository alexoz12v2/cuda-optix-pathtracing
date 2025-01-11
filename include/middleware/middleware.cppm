module;

// this is compilad with cuda too, so needs to be here
// clang-format off
// #include <middleware/middleware-model.h> // not exported bacause part of a CUDA translation unit. include it directly
// clang-format on

#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp>  // glm::pi
#include <glm/mat4x4.hpp>                // glm::mat4
#include <glm/vec3.hpp>                  // glm::vec3
#include <glm/vec4.hpp>                  // glm::vec4

#include <array>
#include <atomic>
#include <forward_list>
#include <limits>
#include <map>
#include <memory_resource>
#include <set>
#include <stack>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>
#include <compare>
#include <cstdint>

export module middleware;

import platform;

#define DMT_NEEDS_MODULE
export
{
#include <middleware/middleware.h>
}

module :private;
#include <middleware.cpp>