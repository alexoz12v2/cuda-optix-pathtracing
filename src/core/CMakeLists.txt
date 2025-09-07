# platform specific implementation files
if(DEFINED DMT_OS_WINDOWS)
  set(core_os_private_sources
    core-texture-cache.win32.cpp
  )
  set(core_unused_sources
    core-texture-cache.linux.cpp
  )
elseif(DEFINED DMT_OS_LINUX)
  set(core_os_private_sources
    core-texture-cache.linux.cpp
  )
  set(core_unused_sources
    core-texture-cache.win32.cpp
  )
endif()
set_source_files_properties(${core_unused_sources} PROPERTIES HEADER_FILE_ONLY TRUE)

## CUDA utils (TODO better and copy to build)
set(cudautils_implementation_files
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils.cu 
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-float.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-vecmath.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-light.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-transform.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-camera.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-media.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-lightsampler.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-texture.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-material.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-sampler.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-film.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-filter.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-bxdf.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-spectrum.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-color.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-numbers.cu
  ${CMAKE_SOURCE_DIR}/src/cudautils/cudautils-image.cu
)
set_source_files_properties(${cudautils_implementation_files} PROPERTIES LANGUAGE CXX)

set(cudautils_header_files
  ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils.h
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
  ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-image.h
)
set_source_files_properties(${cudautils_header_files} PROPERTIES
  LANGUAGE CXX
  HEADER_FILE_ONLY TRUE
)

# target definition
dmt_add_module_library(dmt-core core SHARED)
target_sources(dmt-core
  PUBLIC
    FILE_SET core_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/
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
    FILE_SET cudautils_headers_file_set TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/
      FILES ${cudautils_header_files}
  PRIVATE
    core-parser.cpp
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
    ${core_os_private_sources}
    ${core_unused_sources}
    ${cudautils_implementation_files}
)

target_compile_definitions(dmt-core PRIVATE FBXSDK_SHARED)

target_include_directories(dmt-core 
  PUBLIC
    $<TARGET_PROPERTY:dmt-platform,INTERFACE_INCLUDE_DIRECTORIES>
  PRIVATE 
    ${CMAKE_SOURCE_DIR}/include/cudautils
)

target_link_libraries(dmt-core 
  PUBLIC
    gx::gx
    glm::glm_static
    Eigen3::Eigen
  PRIVATE
    dmt-platform
    stb::stb
    nlohmann_json::nlohmann_json
    FBXSDK::fbxsdk
    # all OpenEXR at the very end
    OpenEXR::OpenEXR
    OpenEXR::OpenEXRCore
    OpenEXR::OpenEXRUtil
    OpenEXR::Iex
    OpenEXR::IlmThread
)

set_target_properties(dmt-core PROPERTIES FOLDER "Modules/Core")

install(TARGETS dmt-core FILE_SET core_headers)

source_group("Unused" FILES ${core_unused_sources})
