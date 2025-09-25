add_library(gx INTERFACE)
target_sources(
  gx
  INTERFACE FILE_SET
            gx_headers
            TYPE
            HEADERS
            BASE_DIRS
            include/
            FILES
            include/gx/shared_mutex.h)
add_library(gx::gx ALIAS gx)
# not needed with install header set target_include_directories(gx INTERFACE include/gx/shared_mutex.h)
install(TARGETS gx FILE_SET gx_headers)
