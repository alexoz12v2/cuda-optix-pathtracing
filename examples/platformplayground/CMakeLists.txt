dmt_add_example(dmt-platform-playground)
target_sources(dmt-platform-playground
  PRIVATE main.cpp stuff.cpp
  PRIVATE FILE_SET dmt_platform_playground_headers TYPE HEADERS FILES stuff.h
)
target_link_libraries(dmt-platform-playground PRIVATE dmt::platform CUDA::cuda_driver dmt::cudautils)
