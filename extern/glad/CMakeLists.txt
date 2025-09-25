include(GenerateExportHeader)
include(GNUInstallDirs)

set(GLAD_HEADER_LIST include/glad/gl.h;include/KHR/khrplatform.h)
set(GLAD_SRC_LIST src/gl.c)
if (DMT_OS_WINDOWS)
  list(APPEND GLAD_HEADER_LIST include/glad/wgl.h)
  list(APPEND GLAD_SRC_LIST src/wgl.c)
endif ()

add_library(glad STATIC)
target_sources(
  glad
  PRIVATE ${GLAD_SRC_LIST}
  PUBLIC FILE_SET
         glad_headers
         TYPE
         HEADERS
         BASE_DIRS
         include/
         FILES
         ${GLAD_HEADER_LIST})

install(TARGETS glad FILE_SET glad_headers)

get_target_property(glad_SOURCES glad INTERFACE_SOURCES)
source_group("GLAD" FILES ${glad_SOURCES})

add_library(glad::glad ALIAS glad)
