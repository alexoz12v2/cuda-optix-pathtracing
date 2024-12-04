dmt_add_example(dmt-example-testdmt
  PRIVATE_SOURCES main.cpp
  PRIVATE_DEPS dmt::testdmt dmt::testdmtpartitions
)
# target_sources(dmt-example-testdmt PRIVATE main.cpp)
# target_link_libraries(dmt-example-testdmt PRIVATE dmt::testdmt)
