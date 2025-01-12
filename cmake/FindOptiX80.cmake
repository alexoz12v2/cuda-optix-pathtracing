# Straight from NVIDIA docs https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/FindOptiX80.cmake
# Looks for the environment variable:
# OPTIX80_PATH

# Sets the variables :
# OPTIX80_INCLUDE_DIR

# OptiX80_FOUND

set(OPTIX80_PATH $ENV{OPTIX80_PATH})

if ("${OPTIX80_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX80_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
  else()
    # Adjust this if the OptiX SDK 8.0.0 installation is in a different location.
    set(OPTIX80_PATH "~/NVIDIA-OptiX-SDK-8.0.0-linux64")
  endif()
endif()

find_path(OPTIX80_INCLUDE_DIR optix_host.h ${OPTIX80_PATH}/include)

# message("OPTIX80_INCLUDE_DIR = " "${OPTIX80_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX80 DEFAULT_MSG OPTIX80_INCLUDE_DIR)

mark_as_advanced(OPTIX80_INCLUDE_DIR)

# Create the target dmtcuda::Optix8
add_library(dmtcuda-Optix8 INTERFACE)
add_library(dmtcuda::Optix8 ALIAS dmtcuda-Optix8)

# Set target properties (adjust as needed)
target_include_directories(dmtcuda-Optix8 
  INTERFACE 
    ${CUDA_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX80_INCLUDE_DIR}
)
target_link_libraries(dmtcuda-Optix8 INTERFACE CUDA::cudart CUDA::cuda_driver)

# Export the target
export(TARGETS dmtcuda-Optix8 FILE "${CMAKE_BINARY_DIR}/dmtcuda-targets.cmake")

# message("OptiX80_FOUND = " "${OptiX80_FOUND}")