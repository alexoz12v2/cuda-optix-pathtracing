dmt_add_example(dmt-example-projtest)
target_sources(dmt-example-projtest PRIVATE main.cpp)

target_link_libraries(dmt-example-projtest PRIVATE dmt::projtest)