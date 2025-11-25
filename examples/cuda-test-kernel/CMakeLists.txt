dmt_add_cli(dmt-test-cuda-kernel)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

target_sources(dmt-test-cuda-kernel PRIVATE ${SOURCES})

target_link_libraries(dmt-test-cuda-kernel PRIVATE dmt::platform stb::stb dmt::core)
