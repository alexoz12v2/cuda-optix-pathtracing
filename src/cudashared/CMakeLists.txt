dmt_add_module_library(dmt-cudashared cudashared SHARED)
target_sources(dmt-cudashared
  PUBLIC
    FILE_SET cudashared_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/cudashared
      FILES ${CMAKE_SOURCE_DIR}/include/cudashared/cudashared.h
            ${CMAKE_SOURCE_DIR}/include/cudashared/cudashared-macros.h
  PRIVATE
    cudashared.cu
)