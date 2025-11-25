# files
dmt_glob_sources(core "${CMAKE_CURRENT_SOURCE_DIR}/private" "${CMAKE_CURRENT_SOURCE_DIR}/public")

# target definition
dmt_add_module_library(dmt-core core)
target_sources(dmt-core
  PUBLIC FILE_SET core_headers TYPE HEADERS BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/public" FILES
  ${HDR_core_FILES}
  PRIVATE
  ${SRC_core_FILES} ${SRC_UNUSED_core_FILES})

target_compile_definitions(dmt-core PRIVATE FBXSDK_SHARED)

set(core_private_dependencies
  glm
  eigen
  stb
  nlohmann_json
  FBXSDK::fbxsdk
)
set(core_public_dependencies
  gx
  dmt-platform
)
foreach (t ${core_private_dependencies} ${core_public_dependencies})
  if (NOT TARGET ${t})
    message(FATAL_ERROR "Target ${t} doesn't exist")
  endif ()
endforeach ()

# they add /EHsc and /MP (Windows, regardless of compiler). remove it
dmt_fix_msvc_interface_compile_options("${core_private_dependencies};${core_public_dependencies}")

target_link_libraries(dmt-core
  PUBLIC ${core_public_dependencies} CUDA::cudart
  PRIVATE ${core_private_dependencies})

