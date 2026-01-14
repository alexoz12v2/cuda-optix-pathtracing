include_guard()

# Reference: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
function(dmt_setup_cuda_toolkit_12_8)
  # we want to control search completely, hence don't use stuff from the system
  # unset(ENV{CUDA_PATH})
  # unset(CACHE{CMAKE_CUDA_COMPILER})
  # unset(CUDACXX)
  # unset(CACHE{CUDAToolkit_BIN_DIR})
  # unset(CACHE{CUDAToolkit_NVCC_EXECUTABLE})
  set(CUDA_VERSION 12.8)
  if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(CUDAToolkit_ROOT "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v${CUDA_VERSION}" CACHE PATH "CUDA Toolkit" FORCE)
    set(CUDAToolkit_BIN_DIR "${CUDAToolkit_ROOT}\\bin" CACHE PATH "CUDA Binary Directory" FORCE)
    set(CUDAToolkit_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}\\nvcc.exe" CACHE PATH "CUDA nvcc Compiler Driver" FORCE)
  elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(CUDAToolkit_ROOT "/usr/local/cuda-${CUDA_VERSION}" CACHE PATH "CUDA Toolkit" FORCE)
    set(CUDAToolkit_BIN_DIR "${CUDAToolkit_ROOT}/bin" CACHE PATH "CUDA Binary Directory" FORCE)
    set(CUDAToolkit_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc" CACHE PATH "CUDA nvcc Compiler Driver" FORCE)
  else ()
    message(FATAL_ERROR "Unsupported OS")
  endif ()
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA nvcc Compiler Driver" FORCE)
  unset(CACHE{CMAKE_CUDA_COMPILER})
  find_program(CMAKE_CUDA_COMPILER
    nvcc
    PATHS "${CUDAToolkit_BIN_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
  #' CUDAToolkit_ROOT)
  find_package(CUDAToolkit ${CUDA_VERSION} REQUIRED EXACT)
  if (NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "Couldn't find CUDA Runtime library target")
  endif ()
  message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
endfunction()
