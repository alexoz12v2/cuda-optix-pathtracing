
#######################################################################################################################
# dmt-training-win32-cooperative-groups
#######################################################################################################################
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

#######################################################################################################################
# dmt-training-test-ewa
#######################################################################################################################
dmt_add_example(dmt-training-test-ewa NO_GUI_WIN32)
dmt_glob_sources(dmt-training-test-ewa "${CMAKE_CURRENT_SOURCE_DIR}/test-ewa" "")
target_sources(dmt-training-test-ewa PRIVATE ${SRC_dmt-training-test-ewa_FILES})
target_include_directories(dmt-training-test-ewa PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test-ewa")
target_link_libraries(dmt-training-test-ewa PRIVATE
  dmt::core
)
add_custom_command(TARGET dmt-training-test-ewa POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/scenes/res/textures/chippedPaint/Paint_Chipped_1K_normal.png" "$<TARGET_FILE_DIR:dmt-training-test-ewa>"
  COMMAND_EXPAND_LISTS
  USES_TERMINAL
)