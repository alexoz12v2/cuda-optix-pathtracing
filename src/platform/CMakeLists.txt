dmt_add_module_library(dmt-platform platform)
target_sources(dmt-platform
  PUBLIC
    FILE_SET platform TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform.cppm
  PUBLIC
    FILE_SET platform_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-logging.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-threadPool.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-memory.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.cuh
  PRIVATE 
    platform-os-utils.h platform-cuda-utils.cu platform-utils.cpp platform-logging.cpp platform-threadPool.cpp platform-memory.cpp platform.cpp
    # platform-os-utils.cpp  # inserted inside .cppm unity build style
)

target_link_libraries(dmt-platform 
  PRIVATE 
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
    CUDA::cudart
    CUDA::cuda_driver
) 