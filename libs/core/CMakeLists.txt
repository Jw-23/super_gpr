add_library(project_core src/optimizer.cpp)
add_library(project::core ALIAS project_core)

target_include_directories(
  project_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(project_core PRIVATE project_options project_warnings)

target_compile_features(project_core PUBLIC cxx_std_14)
target_link_libraries(project_core PUBLIC Eigen3::Eigen PRIVATE project_lbfgspp)
add_library(super_gpr::super_gpr ALIAS project_core)
