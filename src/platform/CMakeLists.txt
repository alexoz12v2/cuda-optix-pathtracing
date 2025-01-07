dmt_add_module_library(dmt-platform platform
  MODULE_INTERFACE platform.cppm
  MODULE_IMPLEMENTATION platform.cpp
  MODULE_PARTITION_INTERFACES platform-utils.cppm platform-logging.cppm platform-threadPool.cppm platform-memory.cppm
  MODULE_PARTITION_IMPLEMENTATIONS platform-utils.cpp platform-logging.cpp platform-threadPool.cpp platform-memory.cpp
  HEADERS platform.h platform-logging.h platform-threadPool.h platform-utils.h platform-memory.h platform-cuda-utils.h platform-cuda-utils.cuh
  PRIVATE_SOURCES platform-os-utils.h platform-os-utils.cpp platform-cuda-utils.cu
)

target_link_libraries(dmt-platform 
  PRIVATE 
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
    CUDA::cudart
    CUDA::cuda_driver
) 