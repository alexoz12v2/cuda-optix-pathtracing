dmt_add_example(dmt-testplot)
target_sources(dmt-testplot
  PRIVATE main.cpp dummy.cu
)
target_link_libraries(dmt-testplot PRIVATE dmt-middleware-mixed implot::implot)
