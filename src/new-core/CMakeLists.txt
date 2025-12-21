# files
dmt_glob_sources(new_core "${CMAKE_CURRENT_SOURCE_DIR}/private" "${CMAKE_CURRENT_SOURCE_DIR}/public")

# target definition
dmt_add_module_library(dmt-new-core core)
target_sources(dmt-new-core
  PUBLIC FILE_SET new_core_headers TYPE HEADERS BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/public" FILES
  ${HDR_new_core_FILES}
  PRIVATE
  ${SRC_new_core_FILES} ${SRC_UNUSED_new_core_FILES})

target_link_libraries(dmt-new-core
  PUBLIC CUDA::cudart)

