dmt_add_example(dmt-exshared
  PRIVATE_SOURCES main.cpp custuff.cu
  PRIVATE_DEPS dmt::testshared dmt-middleware-mixed
)