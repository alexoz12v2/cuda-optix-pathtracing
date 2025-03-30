dmt_add_example(dmt-testbed)
target_sources(dmt-testbed
  PRIVATE main.cpp
)
target_link_libraries(dmt-testbed PRIVATE dmt::application dmt::platform)
