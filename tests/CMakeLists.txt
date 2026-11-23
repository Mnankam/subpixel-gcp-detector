add_executable(gcp_tests
  test_saddle_refiner.cpp
  test_target_detector.cpp
  test_synthetic.cpp
  test_tracker.cpp)
target_link_libraries(gcp_tests PRIVATE gcp gcp_io GTest::gtest_main gcp_options)

include(GoogleTest)
gtest_discover_tests(gcp_tests DISCOVERY_TIMEOUT 60)
