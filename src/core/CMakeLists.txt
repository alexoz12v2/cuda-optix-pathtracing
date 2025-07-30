dmt_add_module_library(dmt-core core SHARED)
target_sources(dmt-core
  PUBLIC
    FILE_SET core_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/core
      FILES
        ${CMAKE_SOURCE_DIR}/include/core/core-macros.h
        ${CMAKE_SOURCE_DIR}/include/core/core-dstd.h
        #${CMAKE_SOURCE_DIR}/include/core/core-parser.h
        ${CMAKE_SOURCE_DIR}/include/core/core-bvh-builder.h
        ${CMAKE_SOURCE_DIR}/include/core/core-primitive.h
        ${CMAKE_SOURCE_DIR}/include/core/core-math.h
        ${CMAKE_SOURCE_DIR}/include/core/core-cudautils-cpubuild.h
        ${CMAKE_SOURCE_DIR}/include/core/core-trianglemesh.h
  PRIVATE
    #core-parser.cpp
    core-bvh-builder.cpp
    core-primitive.cpp
    core-trianglemesh.cpp
    core-math.cpp
    core-cudautils-cpubuild.cpp
)

target_link_libraries(dmt-core 
  PUBLIC
    dmt-platform
    dmt-cudautils
)

set_target_properties(dmt-core PROPERTIES FOLDER "Modules/Core")

install(TARGETS dmt-core FILE_SET core_headers)