#######################################################################################
## cuda core
#######################################################################################
file(GLOB_RECURSE CORE_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/private/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/private/*.cu"
)
file(GLOB_RECURSE CORE_HDRS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/public/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/public/*.cu"
)
dmt_add_module_library(dmt-cuda-core cuda-core cuda-core)
target_sources(dmt-cuda-core
  PUBLIC FILE_SET dmt_cuda_core_headers TYPE HEADERS BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/public" FILES
  ${CORE_HDRS}
  PRIVATE ${CORE_SRCS}
)
target_link_libraries(dmt-cuda-core PUBLIC CUDA::cudart)
target_include_directories(dmt-cuda-core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/public"
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/public/cuda-core"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda-core/private"
)

#######################################################################################
## megakernel executable
#######################################################################################
file(GLOB_RECURSE MEGAKERNEL_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/megakernel/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/megakernel/*.cu"
)
dmt_add_example(dmt-megakernel NO_GUI_WIN32)
target_sources(dmt-megakernel PRIVATE ${MEGAKERNEL_SRCS})
target_link_libraries(dmt-megakernel PRIVATE dmt-cuda-core)
target_include_directories(dmt-megakernel PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/megakernel")

#######################################################################################
## tests
#######################################################################################
## TODO

#######################################################################################
## wavefront executable
#######################################################################################
## TODO
