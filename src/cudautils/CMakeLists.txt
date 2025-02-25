dmt_add_module_library(dmt-cudautils cudautils)
target_sources(dmt-cudautils
  PUBLIC
    FILE_SET cudautils_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/cudautils
      FILES ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-float.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-vecmath.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-enums.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-light.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-transform.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-camera.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-media.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-lightsampler.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-texture.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-material.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-sampler.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-film.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-filter.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-bxdf.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-spectrum.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-color.h
            ${CMAKE_SOURCE_DIR}/include/cudautils/cudautils-numbers.h
  PRIVATE 
    FILE_SET cudautils_private_headers TYPE HEADERS FILES cudautils-vecconv.cuh
  PRIVATE
    cudautils.cu cudautils-float.cu cudautils-vecmath.cu cudautils-light.cu
    cudautils-transform.cu cudautils-camera.cu cudautils-media.cu cudautils-lightsampler.cu
    cudautils-texture.cu cudautils-material.cu cudautils-sampler.cu cudautils-film.cu cudautils-filter.cu
    cudautils-bxdf.cu cudautils-spectrum.cu cudautils-color.cu cudautils-numbers.cu
)
target_include_directories(dmt-cudautils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-cudautils 
  PUBLIC 
    dmt-platform-mixed 
  PRIVATE
    glm::glm
    Eigen3::Eigen
)
source_group("Private Headers" FILES cudautils-vecconv.cuh)