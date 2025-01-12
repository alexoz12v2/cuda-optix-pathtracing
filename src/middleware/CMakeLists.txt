dmt_add_module_library(dmt-middleware middleware)

target_sources(dmt-middleware
  PUBLIC
    FILE_SET middleware TYPE CXX_MODULES BASE_DIRS ${CMAKE_SOURCE_DIR}/include/middleware
      FILES ${CMAKE_SOURCE_DIR}/include/middleware/middleware.cppm
  PUBLIC
    FILE_SET middleware_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/middleware
      FILES ${CMAKE_SOURCE_DIR}/include/middleware/middleware.h
            ${CMAKE_SOURCE_DIR}/include/middleware/middleware-model.h
  PRIVATE
    middleware-model.cu middleware.cpp
)

# TODO I removed clang-tidy
# clang-tidy doens't see modules, so we need to fix that manually (https://discourse.cmake.org/t/c-20-modules-api-example/7303/3)
# get_target_property(DMT_CLANG_TIDY_CMD dmt-middleware-tidy CXX_CLANG_TIDY)
# if(DEFINED DMT_COMPILER_MSVC)
#   list(APPEND DMT_CLANG_TIDY_CMD "--extra-arg=/reference platform=${CMAKE_BINARY_DIR}/src/platform/CMakeFiles/${CMAKE_BUILD_TYPE}/platform.ifc")
#   message(STATUS ${DMT_CLANG_TIDY_CMD})
#   set_target_properties(dmt-middleware-tidy PROPERTIES CXX_CLANG_TIDY "${DMT_CLANG_TIDY_CMD}")
# elseif(DEFINED DMT_COMPILER_GCC OR DEFINED DMT_COMPILER_CLANG)
#   list(APPEND "--extra-arg=-fmodules -fmodule-file=platform=${CMAKE_BINARY_DIR}/src/platform/CMakeFiles/${CMAKE_BUILD_TYPE}/platform.pcm")
#   set_target_properties(dmt-middleware-tidy PROPERTIES CXX_CLANG_TIDY "${DMT_CLANG_TIDY_CMD}")
# endif()

target_link_libraries(dmt-middleware 
  PUBLIC
    dmt-platform-mixed
    glm::glm
  PRIVATE 
    #CUDA::cudart
    #CUDA::cuda_driver
)