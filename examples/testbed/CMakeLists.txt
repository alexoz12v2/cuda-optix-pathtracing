dmt_add_example(dmt-testbed)
target_sources(dmt-testbed
  PRIVATE main.cpp dummy.cu
)
target_link_libraries(dmt-testbed PRIVATE dmt-middleware-mixed)
