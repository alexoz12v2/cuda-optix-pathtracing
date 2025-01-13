dmt_add_example(dmt-testcuda
  PRIVATE_SOURCES main.cpp dummy.cu
  PRIVATE_DEPS dmt-middleware-mixed dmt-cudashared
)