dmt_add_example(dmt-testcuda)
target_sources(dmt-testcuda
  PRIVATE main.cpp dummy.cu
)
target_link_libraries(dmt-testcuda PRIVATE dmt-middleware-mixed)
