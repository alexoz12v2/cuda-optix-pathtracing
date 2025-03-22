dmt_add_example(dmt-exshared)
target_sources(dmt-exshared
  PRIVATE main.cpp custuff.cu
)
target_link_libraries(dmt-exshared PRIVATE dmt-middleware-mixed)
