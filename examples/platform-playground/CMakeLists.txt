dmt_add_example(dmt-platform-playground)
target_sources(dmt-platform-playground PRIVATE main.cpp)
target_link_libraries(dmt-platform-playground PRIVATE dmt::platform dmt::core dmt::cuda-wrappers)
