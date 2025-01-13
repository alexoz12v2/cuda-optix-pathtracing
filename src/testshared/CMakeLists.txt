dmt_add_module_library(dmt-testshared testshared SHARED)
target_sources(dmt-testshared
  PUBLIC
    FILE_SET testshared_headers TYPE HEADERS BASE_DIRS ${CMAKE_SOURCE_DIR}/include/testshared
      FILES ${CMAKE_SOURCE_DIR}/include/testshared/stuff.h
  PRIVATE
    stuff.cpp
)