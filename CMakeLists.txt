cmake_minimum_required(VERSION 3.30) # C++ 20 modules require clang 16, cmake 3.28 at least

################################################################################
# Policies
################################################################################

# Enables the Visibility Property on all target types
# This means you can manage symbol visibility more easily through CMake 
# properties without needing to manually handle compiler flags.
# Usage: This is helpful when building shared libraries, as it lets you define 
# which symbols are exposed externally and which are hidden, 
# improving binary size and load times.
cmake_policy(SET CMP0063 NEW)

# Enables the MSVC_RUNTIME_LIBRARY property on targets
# you can now use the MSVC_RUNTIME_LIBRARY property to specify whether you 
# want to link against the static (/MT) or dynamic (/MD) version of the MSVC 
# runtime instead of having to fiddle with compiler flags on Windows
cmake_policy(SET CMP0091 NEW)

# Enables POSITION_INDEPENDENT_CODE
# with old, you would need to pass -fPIC flag to compilers, on NEW, you can use
# the property POSITION_INDEPENDENT_CODE, depending on the target type 
# (required for Shared Libraries and Executables)
cmake_policy(SET CMP0069 NEW)

# enable new MSVC debug information format specification mechanism if available
# this will embed debug information into the object file to allow ccache to 
# cache it
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

################################################################################
# Project wide Properties and C++ Global Options
################################################################################

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

macro(dmt_set_option var default type docstring)
  if(NOT DEFINED ${var})
    set(${var} ${default})
  endif()
  set(${var} ${${var}} CACHE ${type} ${docstring} FORCE)
endmacro()

# Build Type
dmt_set_option(CMAKE_BUILD_TYPE Debug STRING "Choose the type of build (eg Debug, Release)")

# docs generation
dmt_set_option(DMT_GENERATE_DOCS ON BOOL "Generate Docs")

################################################################################
# Other Global CMake Configuration
################################################################################

project(
  DumbTracer
  VERSION 1.0.0
  DESCRIPTION "Project aiming to perform VR Photon Mapping"
  LANGUAGES CXX C CUDA
)

enable_language(C)
enable_language(CUDA)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Link: https://forums.developer.nvidia.com/t/how-to-include-optix-8-0-in-cmake-project/269123
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

include(cmake/Config.cmake)
include(cmake/Dependencies.cmake)
include(cmake/SetupDoxygen.cmake)

dmt_define_environment()
dmt_setup_dependencies()

# possible TODO: make this work
if(DMT_COMPILER_MSVC)
  dmt_set_option(DMT_GENERATE_PDB ON BOOL "True to generate PDB Symbols")
endif()

dmt_set_option(DMT_ENABLE_SANITIZERS ON BOOL "Enable Sanitizers")
if(DMT_ENABLE_SANITIZERS AND NOT DMT_COMPILER_MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fno-sanitize-recover=all -fsanitize=undefined")
endif()

# output directory of eventual dlls and executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin) # TODO remove, put per target basis

dmt_set_option(CLANG_TIDY_EXECUTABLE clang-tidy STRING "clang-tidy executable program")
include(cmake/ClangTidy.cmake)

# custom target to copy assets from the `res/` directory (use `add_dependencies` in executable targets to have 
# this copy happen automatically)
add_custom_target(dmt-copy-assets 
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/res ${CMAKE_CURRENT_BINARY_DIR}/res
)

# add library code
add_subdirectory(src)

# possible todo: PCH

# build examples
add_subdirectory(examples)

dmt_set_option(DMT_ENABLE_TESTS ON BOOL "ON To turn on tests, OFF to disable them")
if(DMT_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

dmt_set_option(CLANG_FORMAT_EXECUTABLE clang-format STRING "clang-format executable program")
include(cmake/ClangFormat.cmake)

if(DMT_GENERATE_DOCS)
  dmt_setup_doxygen()
endif()