dmt_add_example(dmt-platform-playground)
target_sources(dmt-platform-playground
  PRIVATE main.cpp
)
target_link_libraries(dmt-platform-playground PRIVATE dmt::platform CUDA::cuda_driver dmt::cudautils dmt-cuda-wrappers)
