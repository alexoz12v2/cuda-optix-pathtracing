dmt_add_example(dmt-memtest)
target_sources(dmt-memtest
  PRIVATE main.cpp
)
target_link_libraries(dmt-memtest PRIVATE dmt-platform-mixed)
