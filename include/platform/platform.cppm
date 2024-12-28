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

/**
 * @brief Module `platform`
 */
export module platform;

export import :logging;
export import :threadPool;
export import :memory;
export import :utils; /** Note: Utils contains private, possibly OS specific, functionality + PlatformContext */

export import <platform/platform.h>;

/** @} */
