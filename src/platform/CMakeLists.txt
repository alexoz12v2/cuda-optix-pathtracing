dmt_add_module_library(dmt-platform platform SHARED)
target_sources(dmt-platform
  PUBLIC
    FILE_SET platform_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-logging.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-threadPool.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-memory.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-macros.h
  PRIVATE 
    platform-os-utils.h  platform-utils.cpp platform-logging.cpp platform-threadPool.cpp platform-memory.cpp
    platform-os-utils.cpp  # inserted inside .cppm unity build style
)

target_link_libraries(dmt-platform 
  PUBLIC
    glad 
    glfw 
    imgui 
  PRIVATE 
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
) 

dmt_add_module_library(dmt-platform-cuda platformCuda)
target_sources(dmt-platform-cuda
  PUBLIC
    FILE_SET platform_cuda_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.h 
            ${CMAKE_SOURCE_DIR}/include/platform/platform-cuda-utils.cuh
            ${CMAKE_SOURCE_DIR}/include/platform/cudaTest.h
  PRIVATE
    platform-cuda-utils.cu cudaTest.cu
)
target_link_libraries(dmt-platform-cuda 
  PUBLIC 
    dmt-platform
    #CUDA::cudart # automatically linked
    #CUDA::cuda_driver # automatically linkedb
    dmtcuda-Optix8
)

dmt_add_module_library(dmt-platform-mixed platform)
target_sources(dmt-platform-mixed
  PUBLIC
    FILE_SET platform TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform.cppm
    FILE_SET platform_mized_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/platform
      FILES ${CMAKE_SOURCE_DIR}/include/platform/platform-display.h
            ${CMAKE_SOURCE_DIR}/include/platform/platform.h 
  PRIVATE
    platform-display.cpp platform.cpp
)
target_link_libraries(dmt-platform-mixed 
  PUBLIC 
    dmt-platform-cuda
  PRIVATE
    $<$<BOOL:${DMT_OS_LINUX}>:rt pthread > # needed by async io linux
    $<$<BOOL:${DMT_OS_WINDOWS}>:kernel32.dll advapi32.dll> # needed by fileapi.h (does it automatically with dll, clarity)
    Backward::Backward
)

set_target_properties(dmt-platform-mixed dmt-platform dmt-platform-cuda PROPERTIES FOLDER "Modules/Platform")