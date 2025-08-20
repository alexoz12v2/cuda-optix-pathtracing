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

# target definition
dmt_add_module_library(dmt-core core SHARED)
target_sources(dmt-core
  PUBLIC
    FILE_SET core_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/core
      FILES
        ${CMAKE_SOURCE_DIR}/include/core/core-macros.h
        ${CMAKE_SOURCE_DIR}/include/core/core-dstd.h
        ${CMAKE_SOURCE_DIR}/include/core/core-parser.h
        ${CMAKE_SOURCE_DIR}/include/core/core-bvh-builder.h
        ${CMAKE_SOURCE_DIR}/include/core/core-primitive.h
        ${CMAKE_SOURCE_DIR}/include/core/core-math.h
        ${CMAKE_SOURCE_DIR}/include/core/core-cudautils-cpubuild.h
        ${CMAKE_SOURCE_DIR}/include/core/core-trianglemesh.h
        ${CMAKE_SOURCE_DIR}/include/core/core-bsdf.h
        ${CMAKE_SOURCE_DIR}/include/core/core-light.h
        ${CMAKE_SOURCE_DIR}/include/core/core-light-tree-builder.h
        ${CMAKE_SOURCE_DIR}/include/core/core-texture.h
        ${CMAKE_SOURCE_DIR}/include/core/core-mesh-parser.h
        ${CMAKE_SOURCE_DIR}/include/core/core-texture-cache.h
	${CMAKE_SOURCE_DIR}/include/core/core-material.h
	${CMAKE_SOURCE_DIR}/include/core/core-render.h
  PRIVATE
    core-parser.cpp
    core-bvh-builder.cpp
    core-primitive.cpp
    core-trianglemesh.cpp
    core-math.cpp
    core-cudautils-cpubuild.cpp
    core-bsdf.cpp
    core-light.cpp
    core-light-tree-builder.cpp
    core-texture.cpp
    core-mesh-parser.cpp
    core-material.cpp
    core-render.cpp
    ${core_os_private_sources}
    ${core_unused_sources}
)

target_link_libraries(dmt-core 
  PUBLIC
    dmt-platform
    dmt-cudautils
    gx::gx
  PRIVATE
    FBXSDK::fbxsdk
    stb::stb
    OpenEXR::OpenEXR
    nlohmann_json::nlohmann_json
)

set_target_properties(dmt-core PROPERTIES FOLDER "Modules/Core")

install(TARGETS dmt-core FILE_SET core_headers)

source_group("Unused" FILES ${core_unused_sources})
