/**
 * @file platform-logging.cppm
 * @brief partition interface unit for platform module implementing
 * basic logging functionality. Key features:
 * - logging is performed on 3 statically allocated buffers (stderr, stdout, warning)
 * - such buffers are "leaky", ie they overwrite previous content if they are filled up
 * - logging macros/functions specify what to put on a buffer and then specify a destination, 2 types
 *   - console
 *   - window panel
 *
 * A possibility for the logger to console would be to be asynchronous
 *
 * Desired Usage:
 * - there will be 2 LogDisplays defined: Console Output and Window Panel output
 * - there should be a Macro which decides whether console output logs are turned off or not (compile time log level)
 * - there should be runtime functions to check whether a given log level is enabled or not (runtime log level)
 * @defgroup platform platform Module
 * @{
 */
module;

#include <array>
#include <concepts>
#include <format>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <cassert>
#include <compare>
#include <cstdint>
#include <cstring>

/**
 * @brief module partition `platform:logging`
 */
export module platform:logging;

export import <platform-logging.h>;

/** @} */