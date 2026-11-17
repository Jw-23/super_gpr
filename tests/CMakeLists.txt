add_executable(project_tests test_gpr.cpp)
set_target_properties(project_tests PROPERTIES OUTPUT_NAME "${PROJECT_NAME}_tests")
target_link_libraries(
  project_tests
  PRIVATE GTest::gtest_main project_core project_options project_warnings
)

include(GoogleTest)
gtest_discover_tests(project_tests DISCOVERY_TIMEOUT 60)

option(SUPER_GPR_PYTHON_REFERENCE_TESTS "Compare numerical results with scikit-learn" OFF)
if(SUPER_GPR_PYTHON_REFERENCE_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(sklearn_reference_probe reference_probe.cpp)
  target_link_libraries(sklearn_reference_probe PRIVATE project_core project_options project_warnings)
  add_test(NAME sklearn_reference
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/compare_sklearn.py"
            "$<TARGET_FILE:sklearn_reference_probe>")
endif()
