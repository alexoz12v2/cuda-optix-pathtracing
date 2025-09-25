dmt_add_example(dmt-empty)
target_sources(dmt-empty PRIVATE main.cpp)
target_link_libraries(dmt-empty PRIVATE dmt::platform)
