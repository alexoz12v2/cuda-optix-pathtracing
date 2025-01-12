/**
 * @file platform.cppm
 * @brief Primary interface unit for all utilities necessary to prepare the platform, such as memory allocators,
 * thread pools, ...
 *
 * @defgroup platform platform Module
 * @{
 */

// in the primary module, we need to support both header inclusion and module exports, driven by a macro which
// should be defined by the tranlation unit only when including the interface as an header
module;

// putting everything the .cpp inlined here use, such that their pragma once does its job to prevent double definitions
// while keeping all includes.
// this delicate hack is necessary to let C++20 modules and CUDA talk to each other

// clang-format off
#include <platform/platform-macros.h>
#include <platform/platform-cuda-utils.h> // not exported because its translation unit is a cuda one, so it needs to be included
#include <platform-os-utils.h>
//#include <platform-os-utils.cpp>
#include <platform/cudaTest.h> // used by platform-display.
// clang-format on

#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <compare>
#include <concepts>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits> // std::numeric_limits
#include <numbers>
#include <map>
#include <memory>
#include <mutex>
#include <memory_resource> // std::pmr::memory_resource
#include <queue>
#include <source_location>
#include <shared_mutex>
#include <string_view>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint> // uint32_t, uint64_t, size_t, uintptr_t

// needed by display
#include <glad/gl.h> // before GLFW
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>


/**
 * @brief Module `platform`
 */
export module platform;

#define DMT_NEEDS_MODULE
export
{
#if defined(DMT_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable : 5244)
#endif
#include <platform/platform.h>
}

module :private; // modifying this does not cause recompilation
// clang-format off
#if 0            // these componentgs 
#include <platform-utils.cpp>
#include <platform-logging.cpp>
#include <platform-memory.cpp>
#include <platform-threadPool.cpp>
#include <platform-display.cpp>
#endif
#include <platform.cpp>
// clang-format on

#if defined(DMT_COMPILER_MSVC)
#pragma warning(pop)
#endif

/** @} */
