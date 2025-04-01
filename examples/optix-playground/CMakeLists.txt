dmt_add_example(dmt-optix-playground)
target_sources(dmt-optix-playground
  PRIVATE main.cpp
)
target_link_libraries(dmt-optix-playground PRIVATE dmt::platform dmtcuda-Optix8 dmt::cuda-wrappers)
