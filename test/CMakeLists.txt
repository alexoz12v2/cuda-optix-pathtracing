include(CTest)
include(Catch)

# Build Catch2 in C++17 mode to enable C++17 features
target_compile_features(Catch2 PRIVATE cxx_std_17)

# Ensure that Catch2 sources and headers are not analyzed by any tools
set_target_properties(Catch2 PROPERTIES COMPILE_OPTIONS "" EXPORT_COMPILE_COMMANDS OFF)
set_target_properties(Catch2WithMain PROPERTIES EXPORT_COMPILE_COMMANDS OFF)
get_target_property(CATCH2_INCLUDE_DIRS Catch2 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(Catch2 SYSTEM INTERFACE ${CATCH2_INCLUDE_DIRS})

# base target for tests
add_library(dmt-test-main)
target_sources(dmt-test-main INTERFACE FILE_SET HEADERS FILES shared/dmtutils.h PRIVATE shared/dmtutils.cpp)
target_include_directories(dmt-test-main PUBLIC shared ${PROJECT_SOURCE_DIR}/extern)
target_link_libraries(dmt-test-main INTERFACE Catch2::Catch2WithMain)

# add tests
set(DMT_TEST_TARGETS_LIST "")

# todo: Use OpenCppCoverage to generate coverage report
set(COVERAGE_PREFIX "")

# automatically creates ; lists
dmt_add_test(dmt-test-testdmt
  TARGET_LIST DMT_TEST_TARGETS_LIST
  FILES testdmt/projtest.test.cpp
  DEPENDENCIES dmt-test-main dmt::testdmt
)

dmt_add_test(dmt-test-testdmtpartitions
  TARGET_LIST DMT_TEST_TARGETS_LIST
  FILES testdmtpartitions/testdmtpartitions.test.cpp testdmtpartitions/testdmtpartitions-implonly.test.cpp
        testdmtpartitions/testdmtpartitions-moretrig.test.cpp testdmtpartitions/testdmtpartitions-trig.test.cpp
  DEPENDENCIES dmt-test-main dmt::testdmtpartitions
)

dmt_add_test(dmt-test-platform
  TARGET_LIST DMT_TEST_TARGETS_LIST
  FILES platform/platform.test.cpp platform/platform-logging.test.cpp
  DEPENDENCIES dmt-test-main dmt::platform
)

# run tests as soon as you build them
dmt_set_option(DMT_RUN_TESTS ON BOOL "True to run tests as soon as you build them")
#if(DMT_RUN_TESTS)
#  message(STATUS "Running Tests on targets:${DMT_TEST_TARGETS_LIST}")
#  add_custom_target(dmt-runtests
#    COMMAND ${COVERAGE_PREFIX} ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>
    # COMMAND ${CMAKE_COMMAND} -P "${PROJECT_BINARY_DIR}/patch_coverage.cmake"
#    COMMENT "Run Tests"
#    VERBATIM
#  )
#  message(STATUS "DMT_TEST_TARGETS_LIST: ${DMT_TEST_TARGETS_LIST}")
#  add_dependencies(dmt-runtests ${DMT_TEST_TARGETS_LIST})
#endif()