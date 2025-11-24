add_library(gx INTERFACE)
target_sources(gx
  INTERFACE FILE_SET gx_headers TYPE HEADERS BASE_DIRS include/ FILES include/gx/shared_mutex.h)
add_library(gx::gx ALIAS gx)
target_include_directories(gx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
