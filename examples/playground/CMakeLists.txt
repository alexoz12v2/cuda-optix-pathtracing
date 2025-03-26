dmt_add_example(dmt-playground)
target_sources(dmt-playground
  PRIVATE main.cu printf.cuh printf.cu
)
target_link_libraries(dmt-playground PRIVATE dmt-platform glfw glad::glad)
