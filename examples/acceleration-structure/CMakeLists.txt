dmt_add_example(dmt-acceleration-structure)

target_sources(dmt-acceleration-structure
  PRIVATE FILE_SET utilites_headers TYPE HEADERS FILES utilities.h
  PRIVATE main.cpp utilities.cpp
)
target_include_directories(dmt-acceleration-structure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dmt-acceleration-structure PRIVATE dmt::platform dmt::core)
