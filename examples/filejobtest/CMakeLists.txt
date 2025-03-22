dmt_add_example(dmt-filejobtest)
target_sources(dmt-filejobtest
  PRIVATE main.cpp cuda-tests.h cuda-tests.cu
)
target_link_libraries(dmt-filejobtest PRIVATE dmt-middleware-mixed)
