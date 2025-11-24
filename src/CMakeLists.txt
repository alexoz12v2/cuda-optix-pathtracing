add_subdirectory(platform)

# We are removing manual CUDA Fetching. We expect the user to have CUDA 11.8 Toolkit installed
# add_subdirectory(cuda-wrappers)

add_subdirectory(core)

# we are not building a GUI Application anymore
# add_subdirectory(application)
