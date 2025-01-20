dmt_add_example(dmt-testmapunified
  PRIVATE_SOURCES main.cu
  PRIVATE_DEPS implot::implot dmt-platform-mixed dmt::cudautils
)