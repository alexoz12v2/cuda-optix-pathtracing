dmt_add_cli(dmt-test-cuda)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

target_sources(dmt-test-cuda PRIVATE ${SOURCES})
target_include_directories(dmt-test-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-test-cuda PRIVATE dmt::platform dmt::core stb::stb OpenEXR::OpenEXR)
