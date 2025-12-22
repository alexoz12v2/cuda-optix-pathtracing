file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

dmt_add_example(dmt-testbed-2 NO_GUI_WIN32)
target_sources(dmt-testbed-2 PRIVATE ${SRCS})
target_link_libraries(dmt-testbed-2 PRIVATE dmt::new-core)

