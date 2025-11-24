file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

dmt_add_example(dmt-cuda-test-2 NO_GUI_WIN32)
target_sources(dmt-cuda-test-2 PRIVATE ${SRCS})
target_link_libraries(dmt-cuda-test-2 PRIVATE CUDA::cudart)
