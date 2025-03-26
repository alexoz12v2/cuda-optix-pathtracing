dmt_add_module_library(dmt-middleware middleware)# SHARED) cannot be shared if depends and exposes CUDA code (PUBLIC cuda dependendies)
target_sources(dmt-middleware
  PUBLIC
    FILE_SET middleware_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/middleware
      FILES ${CMAKE_SOURCE_DIR}/include/middleware/middleware-utils.h
            ${CMAKE_SOURCE_DIR}/include/middleware/middleware-parser.h
  PRIVATE
    middeleware-utils.cpp middleware-parser.cpp
)

target_link_libraries(dmt-middleware 
  PUBLIC
    dmt-platform
    dmt-cudautils
)

dmt_add_module_library(dmt-middleware-cuda middleware)
target_sources(dmt-middleware-cuda
  PUBLIC
    FILE_SET middleware_cuda_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/middleware
      FILES ${CMAKE_SOURCE_DIR}/include/middleware/middleware-model.h
  PRIVATE
    middleware-model.cu 
)
target_link_libraries(dmt-middleware-cuda PUBLIC dmt-platform)

dmt_add_module_library(dmt-middleware-mixed middleware)
target_sources(dmt-middleware-mixed
  PUBLIC
    FILE_SET middleware_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/middleware
      FILES ${CMAKE_SOURCE_DIR}/include/middleware/middleware.h
            
  PRIVATE
    middleware.cpp
)
target_link_libraries(dmt-middleware-mixed PUBLIC dmt-middleware dmt-middleware-cuda)
set_target_properties(dmt-middleware-mixed dmt-middleware dmt-middleware-cuda PROPERTIES FOLDER "Modules/Middleware")