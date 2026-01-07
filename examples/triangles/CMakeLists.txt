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
target_link_libraries(dmt-cuda-core PUBLIC CUDA::cudart PRIVATE stb)
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
## megakernel executable
#######################################################################################
file(GLOB_RECURSE WAVEFRONT_STREAM_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/wavefront-stream/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/wavefront-stream/*.cu"
)
dmt_add_example(dmt-wavefront-stream NO_GUI_WIN32)
target_sources(dmt-wavefront-stream PRIVATE ${WAVEFRONT_STREAM_SRCS})
target_link_libraries(dmt-wavefront-stream PRIVATE dmt-cuda-core)
target_include_directories(dmt-wavefront-stream PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/wavefront-stream")

#######################################################################################
## tests
#######################################################################################

## BVH Experimentation Test
file(GLOB_RECURSE TEST_BVH_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/bvh/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/bvh/*.cu"
)
dmt_add_example(dmt-test-bvh NO_GUI_WIN32)
target_sources(dmt-test-bvh PRIVATE ${TEST_BVH_SRCS})
target_link_libraries(dmt-test-bvh PRIVATE dmt-cuda-core)
target_include_directories(dmt-test-bvh PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/bvh")

## Queue Test
file(GLOB_RECURSE TEST_QUEUE_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/queue/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/queue/*.cu"
)
dmt_add_example(dmt-test-queue NO_GUI_WIN32)
target_sources(dmt-test-queue PRIVATE ${TEST_QUEUE_SRCS})
target_link_libraries(dmt-test-queue PRIVATE dmt-cuda-core)
target_include_directories(dmt-test-queue PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/queue/")

## TODO

#######################################################################################
## wavefront executable
#######################################################################################
file(GLOB_RECURSE WAVEFRONT_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/wavefront/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/wavefront/*.cu"
)
dmt_add_example(dmt-wavefront NO_GUI_WIN32)
target_sources(dmt-wavefront PRIVATE ${WAVEFRONT_SRCS})
target_link_libraries(dmt-wavefront PRIVATE dmt-cuda-core)
target_include_directories(dmt-wavefront PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/wavefront")
