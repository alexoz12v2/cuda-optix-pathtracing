#include "cudashared.h"

#include <cuda.h>

namespace dmt::test {
    __host__ __device__ float multiply(float a, float b) { return a * b; }
} // namespace dmt::test