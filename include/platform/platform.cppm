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

// clang-format off
// #include <platform/platform-cuda-utils.h> // not exported because its translation unit is a cuda one, so it needs to be included
#include <platform-os-utils.h>
#include <platform-os-utils.cpp>
// clang-format on

#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <iterator>
#include <limits> // std::numeric_limits
#include <map>
#include <memory>
#include <memory_resource> // std::pmr::memory_resource
#include <queue>
#include <source_location>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <compare>
//
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint> // uint32_t, uint64_t, size_t, uintptr_t

/**
 * @brief Module `platform`
 */
export module platform;

#define DMT_NEEDS_MODULE
export
{
#include <platform/platform.h>
}

module :private; // modifying this does not cause recompilation
// clang-format off
#include <platform-logging.cpp>
#include <platform-memory.cpp>
#include <platform-threadPool.cpp>
#include <platform-utils.cpp>
#include <platform.cpp>
// clang-format on

/** @} */
