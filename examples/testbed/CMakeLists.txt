file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

dmt_add_example(dmt-testbed NO_GUI_WIN32)
target_sources(dmt-testbed PRIVATE ${SRCS})
target_link_libraries(dmt-testbed PRIVATE dmt::new-core)

