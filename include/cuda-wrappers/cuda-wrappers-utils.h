#pragma once

#include "cuda-wrappers/cuda-wrappers-macros.h"
#include "cuda-wrappers/cuda-wrappers-cuda-driver.h"  // includes <cuda.h>

namespace dmt {
/**
 * Checks the `CUresult` of a CUDA Driver API operation, and, if different than
 * `::CUDA_SUCCESS`,
 * @note This is supposed to be used in CUDA calls whose failure is fatal
 */
[[nodiscard]] DMT_CUDA_WRAPPERS_API bool cudaDriverCall(
    CUDADriverLibrary const* cudaApi, CUresult result);

/**
 * This function (and similiar for all loaded dlls) should be populated with
 * more manual fixes with respect to the generated version as soon as Access
 * Violations are discovered
 */
DMT_CUDA_WRAPPERS_API void fixCUDADriverSymbols(CUDADriverLibrary* cudaApi);
}  // namespace dmt