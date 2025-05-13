dmt_add_example(dmt-acceleration-structure)

target_sources(dmt-acceleration-structure
  PRIVATE main.cpp
)
target_link_libraries(dmt-acceleration-structure PRIVATE dmt::platform dmt::cudautils)
