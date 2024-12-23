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

#include <cstdint>

/**
 * @brief Module `platform`
 */
export module platform;
export import :threadPool;
export import :logging;
export import :display

export import <platform.h>;

/** @} */