#pragma once

#include <cudashared/cudashared-macros.h>

namespace dmt::test {
    DMT_CUDASHARED_API DMT_CPU_GPU float multiply(float a, float b);
}