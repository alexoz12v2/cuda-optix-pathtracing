#ifndef DMT_TRAINING_COOPERATIVE_GROUPS_EXAMPLE_H
#define DMT_TRAINING_COOPERATIVE_GROUPS_EXAMPLE_H

#include "the-macros.h"

// windows
#include "combaseapi.h"

namespace dmt::win32 {
    [[noreturn]] void printResultAndExitProcess(HRESULT result);
}

namespace dmt::_1basics {

    void printCudaCapableDevices();

    void saxpy(float const* __restrict v0, float const* __restrict v1, float* v2, float a, size_t count);

    void matrixSum(float const* __restrict A, float const* __restrict B, size_t rows, size_t cols, float* __restrict result);

} // namespace dmt::_1basics

#endif // DMT_TRAINING_COOPERATIVE_GROUPS_EXAMPLE_H
