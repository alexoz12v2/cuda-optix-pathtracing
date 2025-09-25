# platform specific implementation files
if (DEFINED DMT_OS_WINDOWS)
  set(core_os_private_sources core-texture-cache.win32.cpp)
  set(core_unused_sources core-texture-cache.linux.cpp)
elseif (DEFINED DMT_OS_LINUX)
  set(core_os_private_sources core-texture-cache.linux.cpp)
  set(core_unused_sources core-texture-cache.win32.cpp)
endif ()
set_source_files_properties(${core_unused_sources} PROPERTIES HEADER_FILE_ONLY TRUE)

# CUDA utils (TODO better and copy to build)

set(cudautils_header_files
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-macro.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-float.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-vecmath.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-enums.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-light.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-transform.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-camera.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-media.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-lightsampler.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-texture.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-material.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-sampler.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-film.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-filter.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-bxdf.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-spectrum.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-color.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-numbers.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-bvh.h
    ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-pools.h)

set_source_files_properties(${cudautils_header_files} PROPERTIES LANGUAGE CXX HEADER_FILE_ONLY TRUE)

source_group("CUDA Files" FILES ${cudautils_header_files})

add_custom_target(dmt-copy-cudautils ALL COMMENT "Copy cudautils files")

# custom target to copy shaders for runtime kernel compilation
add_custom_command(
  TARGET dmt-copy-cudautils
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/cudautils)

foreach (shader_file IN LISTS cudautils_header_files)
  add_custom_command(
    TARGET dmt-copy-cudautils
    COMMAND ${CMAKE_COMMAND} -E copy ${shader_file} ${CMAKE_BINARY_DIR}/bin/shaders/cudautils
    COMMENT "Copying ${shader_file}")
endforeach ()
add_custom_command(
  TARGET dmt-copy-cudautils
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/include/dmtmacros.h ${CMAKE_BINARY_DIR}/bin/shaders
  COMMENT "copying macros file")

# target definition
dmt_add_module_library(dmt-core core SHARED)
target_sources(
  dmt-core
  PUBLIC FILE_SET
         core_headers
         TYPE
         HEADERS
         BASE_DIRS
         ${CMAKE_SOURCE_DIR}/include/
         FILES
         ${CMAKE_SOURCE_DIR}/include/core/core-macros.h
         ${CMAKE_SOURCE_DIR}/include/core/core-dstd.h
         ${CMAKE_SOURCE_DIR}/include/core/core-parser.h
         ${CMAKE_SOURCE_DIR}/include/core/core-bvh-builder.h
         ${CMAKE_SOURCE_DIR}/include/core/core-primitive.h
         ${CMAKE_SOURCE_DIR}/include/core/core-math.h
         ${CMAKE_SOURCE_DIR}/include/core/core-trianglemesh.h
         ${CMAKE_SOURCE_DIR}/include/core/core-bsdf.h
         ${CMAKE_SOURCE_DIR}/include/core/core-light.h
         ${CMAKE_SOURCE_DIR}/include/core/core-light-tree-builder.h
         ${CMAKE_SOURCE_DIR}/include/core/core-texture.h
         ${CMAKE_SOURCE_DIR}/include/core/core-mesh-parser.h
         ${CMAKE_SOURCE_DIR}/include/core/core-texture-cache.h
         ${CMAKE_SOURCE_DIR}/include/core/core-material.h
         ${CMAKE_SOURCE_DIR}/include/core/core-render.h
         FILE_SET
         cudautils_headers_file_set
         TYPE
         HEADERS
         BASE_DIRS
         ${CMAKE_SOURCE_DIR}/include/
         FILES
         ${cudautils_header_files}
  PRIVATE core-parser.cpp
          core-bvh-builder.cpp
          core-primitive.cpp
          core-trianglemesh.cpp
          core-math.cpp
          core-bsdf.cpp
          core-light.cpp
          core-light-tree-builder.cpp
          core-texture.cpp
          core-mesh-parser.cpp
          core-material.cpp
          core-render.cpp
          core-texture-cache.cpp
          core-cudautils-cpubuild.cpp
          ${core_os_private_sources}
          ${core_unused_sources})

target_compile_definitions(dmt-core PRIVATE FBXSDK_SHARED)

target_include_directories(
  dmt-core
  PUBLIC $<TARGET_PROPERTY:dmt-platform,INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE ${CMAKE_SOURCE_DIR}/include/cudautils)

target_link_libraries(
  dmt-core
  PUBLIC gx::gx
         # CUDA UTILS LIBARIES (together with platform?)
         dmt::cuda-wrappers glm::glm_static Eigen3::Eigen
  PRIVATE dmt-platform
          stb::stb
          nlohmann_json::nlohmann_json
          FBXSDK::fbxsdk
          # all OpenEXR at the very end
          OpenEXR::OpenEXR
          OpenEXR::OpenEXRCore
          OpenEXR::OpenEXRUtil
          OpenEXR::Iex
          OpenEXR::IlmThread)

set_target_properties(dmt-core PROPERTIES FOLDER "Modules/Core")

install(TARGETS dmt-core FILE_SET core_headers)

source_group("Unused" FILES ${core_unused_sources})
