dmt_add_example(dmt-playground
  PRIVATE_SOURCES main.cu printf.cuh printf.cu
  PRIVATE_DEPS dmt-platform-mixed glfw glad::glad
)
