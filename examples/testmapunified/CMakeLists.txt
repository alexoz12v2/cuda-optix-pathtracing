dmt_add_example(dmt-testmapunified)
target_sources(dmt-testmapunified
  PRIVATE main.cu
)
target_link_libraries(dmt-testmapunified PRIVATE implot::implot dmt::platform dmt::cudautils)
