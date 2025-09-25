dmt_add_example(dmt-exshared)
target_sources(dmt-exshared PRIVATE main.cpp)
target_link_libraries(dmt-exshared PRIVATE dmt::platform)
