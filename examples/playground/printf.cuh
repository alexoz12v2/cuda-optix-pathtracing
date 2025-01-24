#pragma once

#include "dmtmacros.h"

#include <cstdarg>

// Source: https://github.com/eyalroz/cuda-kat/blob/development/src/kat/on_device/c_standard_library/printf.cu
namespace dmt {
    DMT_GPU int snprintf(char* s, size_t count, char const* format, ...);

    DMT_GPU int vsnprintf(char* s, size_t count, char const* format, va_list arg);
} // namespace dmt
