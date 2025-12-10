
# Cooperative Groups training example
dmt_add_example(dmt-training-win32-cooperative-groups NO_GUI_WIN32)
dmt_glob_sources(win32_cooperative_groups "${CMAKE_CURRENT_SOURCE_DIR}/cooperative-groups" "")
target_sources(dmt-training-win32-cooperative-groups PRIVATE ${SRC_win32_cooperative_groups_FILES})
target_include_directories(dmt-training-win32-cooperative-groups PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cooperative-groups")
target_link_libraries(dmt-training-win32-cooperative-groups PRIVATE
  dmt::core
  CUDA::cudart
  CUDA::cuda_driver
  Ole32 # combaseapi.h/Objbase.h --> CoInitializeEx
)
