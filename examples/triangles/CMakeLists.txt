file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

dmt_add_example(dmt-test-triangle NO_GUI_WIN32)
target_sources(dmt-test-triangle PRIVATE ${SRCS})
target_link_libraries(dmt-test-triangle PRIVATE CUDA::cudart)
target_include_directories(dmt-test-triangle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
