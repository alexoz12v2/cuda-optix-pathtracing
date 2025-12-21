#pragma once

#include "cuda-wrappers/cuda-wrappers-cuda-driver.h"
#include "cuda-wrappers/cuda-wrappers-utils.h"
#include "cuda-wrappers/cuda-wrappers-nvrtc.h"
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstddef>
#include "core/core-math.h"
#include "core/core-render.h"

namespace dmt {

std::vector<char const*> getnvccOpts(bool debug);

// kernelFileName must be '\0' terminated
std::unique_ptr<char[]> compilePTX(dmt::os::Path const& path,
                                   NVRTCLibrary* nvrtcApi,
                                   std::string_view kernelFileName,
                                   std::vector<char const*> const& nvccOpts);
std::unique_ptr<char[]> compilePTX(std::string_view srcKernel,
                                   NVRTCLibrary* nvrtcApi,
                                   std::string_view kernelFileName,
                                   std::vector<char const*> const& nvccOpts);
GpuSamplerHandle uploadFilterDistrib(CUDADriverLibrary* cudaApi,
                                     PiecewiseConstant2D const& cpuDistrib,
                                     filtering::Mitchell const& cpuFilter);
PiecewiseConstant2D precalculateMitchellDistrib(
    filtering::Mitchell const& filter, int Nx, int Ny,
    std::pmr::memory_resource* mem = std::pmr::get_default_resource());
}  // namespace dmt