include(GenerateExportHeader)
include(GNUInstallDirs)

add_library(glad STATIC)
target_sources(glad
  PRIVATE
    src/gl.c
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/glad/gl.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/glad/gl.h>

    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/glad/wgl.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/glad/wgl.h>

    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/KHR/khrplatform.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/KHR/khrplatform.h>)

target_include_directories(glad
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(glad::glad ALIAS glad)