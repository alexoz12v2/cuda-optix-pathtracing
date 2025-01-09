dmt_add_module_library(dmt-middleware middleware
  MODULE_INTERFACE middleware.cppm
  MODULE_PARTITION_INTERFACES middleware-model.cppm
  MODULE_IMPLEMENTATION middleware.cpp
  HEADERS middleware.h middleware-model.h
  PRIVATE_SOURCES middleware-model.cu
)

target_link_libraries(dmt-middleware 
  PUBLIC
    dmt::platform
    glm::glm
  PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)