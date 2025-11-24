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

target_link_libraries(dmt-core
  PUBLIC gx::gx dmt-platform
  PRIVATE
  glm::glm
  Eigen3::Eigen
  stb::stb
  nlohmann_json::nlohmann_json
  FBXSDK::fbxsdk
  OpenEXR::OpenEXR
  OpenEXR::OpenEXRCore
  OpenEXR::OpenEXRUtil
  OpenEXR::Iex
  OpenEXR::IlmThread)
