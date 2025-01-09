dmt_add_example(dmt-filejobtest 
  PRIVATE_SOURCES main.cpp cuda-tests.h cuda-tests.cu
  PRIVATE_DEPS dmt::platform dmt::middleware CUDA::cudart CUDA::cuda_driver
)