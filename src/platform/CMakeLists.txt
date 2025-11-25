# --- Targets ---
dmt_add_module_library(dmt-platform platform)
dmt_glob_sources(platform "${CMAKE_CURRENT_SOURCE_DIR}/private" "${CMAKE_CURRENT_SOURCE_DIR}/public")

# --- dmt-platform target ---
target_sources(
  dmt-platform
  PUBLIC FILE_SET platform_headers TYPE HEADERS BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/public" FILES
  ${HDR_platform_FILES}
  PRIVATE
  ${SRC_platform_FILES} ${SRC_UNUSED_platform_FILES})
dmt_fix_msvc_interface_compile_options(backward)
target_link_libraries(dmt-platform PRIVATE
  $<$<BOOL:${DMT_OS_LINUX}>:rt pthread>
  $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll>
  backward)
