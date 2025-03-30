dmt_add_example(dmt-testplot)
target_sources(dmt-testplot
  PRIVATE main.cpp
)
target_link_libraries(dmt-testplot PRIVATE dmt::platform implot::implot dmt::cudautils)
