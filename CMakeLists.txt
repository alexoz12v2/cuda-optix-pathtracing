cmake_minimum_required(VERSION 4.2.1) # C++ 20 modules require clang 16, cmake 3.28 at least

# ######################################################################################################################
# Policies
# ######################################################################################################################

# Enables the Visibility Property on all target types This means you can manage symbol visibility more easily through
# CMake properties without needing to manually handle compiler flags. Usage: This is helpful when building shared
# libraries, as it lets you define which symbols are exposed externally and which are hidden, improving binary size and
# load times.
cmake_policy(SET CMP0063 NEW)

# Enables the MSVC_RUNTIME_LIBRARY property on targets you can now use the MSVC_RUNTIME_LIBRARY property to specify
# whether you want to link against the static (/MT) or dynamic (/MD) version of the MSVC runtime instead of having to
# fiddle with compiler flags on Windows
cmake_policy(SET CMP0091 NEW)

# Enables POSITION_INDEPENDENT_CODE with old, you would need to pass -fPIC flag to compilers, on NEW, you can use the
# property POSITION_INDEPENDENT_CODE, depending on the target type (required for Shared Libraries and Executables)
cmake_policy(SET CMP0069 NEW)

# enable new MSVC debug information format specification mechanism if available this will embed debug information into
# the object file to allow ccache to cache it
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif ()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

project(
  DumbTracer
  VERSION 1.0.0
  DESCRIPTION "CUDA Path Tracing"
)
enable_language(CUDA)
## message(FATAL_ERROR
##   "C++ compiler         : ${CMAKE_CXX_COMPILER}\n"
##   "C++ linker executable:  ${CMAKE_CXX_LINK_EXECUTABLE} \n"
##   "CUDA compiler driver : ${CMAKE_CUDA_COMPILER}"
## )
if (NOT TARGET CUDA::cudart)
  include("cmake/CUDA.toolchain.cmake")
  dmt_setup_cuda_toolkit_12_8()
endif()

# ######################################################################################################################
# Project wide Properties and C++ Global Options
# ######################################################################################################################
macro(dmt_set_option var default type docstring)
  if (NOT DEFINED ${var})
    set(${var} ${default})
  endif ()
  set(${var}
    ${${var}}
    CACHE ${type} ${docstring} FORCE)
endmacro()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set the project version (used for versioning and CPack)
set(CMAKE_PROJECT_VERSION_MAJOR 1)
set(CMAKE_PROJECT_VERSION_MINOR 0)
set(CMAKE_PROJECT_VERSION_PATCH 0)
set(CMAKE_PROJECT_VERSION ${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH})

# Set required C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)


dmt_set_option(DMT_NVCC_MAXREGCOUNT "32" "STRING" "nvcc --maxregcount")
dmt_set_option(DMT_DEVICE_LINK_TIME_OPTIMIZATION OFF BOOL "-dlto to nvcc and nvlink")
dmt_set_option(DMT_NVCC_ARCHITECTURES "61-real" STRING "physical architecture for gencode when NOT using -dlto to ptxas and nvlink")
dmt_set_option(DMT_NVCC_LTO_VERSION "61" STRING "physical architecture number for arch when using -dlto to ptxas and nvlink")
dmt_set_option(CMAKE_BUILD_TYPE Debug STRING "Choose the type of build (eg Debug, Release)")
dmt_set_option(DMT_GENERATE_DOCS OFF BOOL "Generate Docs")
dmt_set_option(DMT_GENERATE_TRAINING OFF BOOL "Generate Training Targets")

## Device Link Time Optimization not supported on Windows. not nice.
## https://forums.developer.nvidia.com/t/is-clang-cl-officially-supported-as-a-host-compiler-on-windows/278397/3
if (DMT_DEVICE_LINK_TIME_OPTIMIZATION)
  message(STATUS
    "CMAKE_CXX_COMPILER_FRONTEND_VARIANT: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}\n"
    "and compiler is ID: ${CMAKE_CXX_COMPILER_ID}\n"
    "and linker CMAKE_CXX_COMPILER_LINKER_FRONTEND_VARIANT: ${CMAKE_CXX_COMPILER_LINKER_FRONTEND_VARIANT}"
  )
  include(CheckIPOSupported)
  check_ipo_supported(RESULT YES_OR_NO OUTPUT THE_OUTPUT LANGUAGES CXX CUDA)
  if (NOT YES_OR_NO STREQUAL "YES")
    message(FATAL_ERROR "check_ipo_supported: ${THE_OUTPUT}")
  else ()
    message(STATUS "check_ipo_supported: ${THE_OUTPUT}")
  endif ()
  set(CMAKE_CUDA_ARCHITECTURES ${DMT_NVCC_LTO_VERSION})
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
  set(DMT_DEVICE_LINK_TIME_OPTIMIZATION OFF CACHE BOOL "" FORCE)
endif ()

include(cmake/Config.cmake)
include(cmake/Dependencies.cmake)
include(cmake/SetupDoxygen.cmake)

dmt_define_environment()

### TO BE PUT INTO TOOLCHAIN
#### include("${CMAKE_SOURCE_DIR}/cmake/CUDA.toolchain.cmake")
#### dmt_setup_cuda_toolkit_12_8()
####
#### # Enable CUDA After including toolkit
#### if (WIN32)
####   message(STATUS "Printing Include environemnt variable")
####   message(STATUS "$ENV{INCLUDE}")
#### endif ()
#### enable_language(CUDA)

# Dependencies after CUDA setup
dmt_setup_dependencies()

# grab python executable (msut be after `dmt_define_environment`). sets variable PYTHON_EXEC
dmt_find_python_executable()
block()
  execute_process(COMMAND ${PYTHON_EXEC} ${PROJECT_SOURCE_DIR}/scripts/test_python.py COMMAND_ECHO STDOUT
    OUTPUT_VARIABLE test_python_STDOUT)
  message(STATUS "${test_python_STDOUT}")
endblock()

# possible TODO: make this work
if (DMT_COMPILER_MSVC)
  dmt_set_option(DMT_GENERATE_PDB ON BOOL "True to generate PDB Symbols")
endif ()

dmt_set_option(DMT_ENABLE_SANITIZERS ON BOOL "Enable Sanitizers")
if (DMT_ENABLE_SANITIZERS AND NOT DMT_COMPILER_MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fno-sanitize-recover=all -fsanitize=undefined")
endif ()

# output directory of eventual dlls and executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin) # TODO remove, put per target basis

dmt_set_option(CLANG_TIDY_EXECUTABLE clang-tidy STRING "clang-tidy executable program")
include(cmake/ClangTidy.cmake)

# custom target to copy assets from the `res/` directory (use `add_dependencies` in executable targets to have this copy
# happen automatically)
add_custom_target(
  dmt-copy-assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/res ${CMAKE_CURRENT_BINARY_DIR}/res
  COMMENT "copying in the build directory the assets directory")

add_custom_target(
  dmt-link-assets
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/res ${CMAKE_CURRENT_BINARY_DIR}/res
  COMMENT "Creating symbolic link for assets directory")

# Declare targets to Copy CUDA Device code inside the build/shaders directory
# add_custom_target(dmt-copy-cuda-device-code ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPTIX80_PATH}/include
# ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/shaders COMMAND ${CMAKE_COMMAND} -E copy_directory
# ${OPTIX80_PATH}/include/internal ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/shaders/internal COMMENT "Copy CUDA
# Device code source header files for JIT compilation" )

add_custom_target(
  dmt-copy-cuda-dlls ALL
  COMMAND ${PYTHON_EXEC} ${PROJECT_SOURCE_DIR}/scripts/copy_dlls.py ${CMAKE_BINARY_DIR}/bin
  COMMENT "Copy DLLs to bin")

set(dmt_cuda_toolkit_include_dir $ENV{CUDA_HOME}/include)

# add library code
add_subdirectory(src)

# possible todo: PCH

# build examples
add_subdirectory(examples)
## add_subdirectory(cli)

## dmt_set_option(DMT_ENABLE_TESTS OFF BOOL "ON To turn on tests, OFF to disable them")
## if (DMT_ENABLE_TESTS)
##   enable_testing()
##   add_subdirectory(test)
## endif ()

dmt_set_option(CLANG_FORMAT_EXECUTABLE clang-format STRING "clang-format executable program")
include(cmake/ClangFormat.cmake)

## if (DMT_GENERATE_DOCS)
##   dmt_setup_doxygen()
## endif ()

## if (DMT_GENERATE_TRAINING)
##   add_subdirectory(training)
## endif ()
