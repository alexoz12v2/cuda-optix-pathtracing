include(GenerateExportHeader)
include(GNUInstallDirs)

add_library(glad STATIC)
target_sources(glad
  PRIVATE
    src/gl.c
    src/wgl.c
  PUBLIC 
    FILE_SET glad_headers TYPE HEADERS BASE_DIRS include/
      FILES include/glad/gl.h
            include/glad/wgl.h
            include/KHR/khrplatform.h
)

target_include_directories(glad
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

get_target_property(glad_SOURCES glad INTERFACE_SOURCES)
source_group("GLAD" FILES ${glad_SOURCES})

add_library(glad::glad ALIAS glad)