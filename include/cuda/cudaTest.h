#pragma once

#include <cstdint>

namespace dmt
{

void kernel(float const* A, float const* B, float scalar, float* C, uint32_t N);

}