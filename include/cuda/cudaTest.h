#pragma once

#include <cstdint>

namespace dmt
{

uint32_t createOpenGLTexture(int width, int height);
bool     RegImg(uint32_t tex, uint32_t width, uint32_t height);
void     kernel(float const* A, float const* B, float scalar, float* C, uint32_t N);

} // namespace dmt