/**
 * @file platform.cppm
 * @brief Primary interface unit for all utilities necessary to prepare the platform, such as memory allocators,
 * thread pools, ...
 *
 * @defgroup platform platform Module
 * @{
 */
module;

/**
 * @brief Module `platform`
 */
export module platform;

export import :logging;
export import :memory;
export import :utils; /** Note: Utils contains private, possibly OS specific, functionality + PlatformContext */

export import <platform.h>

/** @} */
