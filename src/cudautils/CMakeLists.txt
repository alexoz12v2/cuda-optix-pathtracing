dmt_add_module_library(dmt-cudautils cudautils)
target_sources(dmt-cudautils
  PUBLIC
    FILE_SET cudautils_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/cudautils
      FILES ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-float.h
  PRIVATE
    cudautils.cu cudautils-float.cu
)
target_link_libraries(dmt-cudautils 
  PUBLIC 
    dmt-platform-mixed 
    glm::glm
)