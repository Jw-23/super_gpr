include("${CMAKE_CURRENT_LIST_DIR}/../CPM.cmake")

if(CMAKE_CXX_STANDARD LESS 17)
  # These are the newest release lines that build with a C++14-only compiler.
  set(PROJECT_GOOGLETEST_VERSION 1.16.0)
  set(PROJECT_BENCHMARK_VERSION 1.9.0)
else()
  set(PROJECT_GOOGLETEST_VERSION 1.18.0)
  set(PROJECT_BENCHMARK_VERSION 1.9.5)
endif()

if(PROJECT_ENABLE_TESTING)

  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    VERSION ${PROJECT_GOOGLETEST_VERSION}
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
  )
endif()

if(PROJECT_ENABLE_BENCHMARKS)
  CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION ${PROJECT_BENCHMARK_VERSION}
    OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
  )
endif()

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  CPMAddPackage(
    NAME eigen
    URL https://github.com/eigen-mirror/eigen/archive/refs/tags/3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_ONLY YES
  )
  add_library(project_eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS project_eigen)
  target_include_directories(project_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
endif()

add_library(project_lbfgspp INTERFACE)
target_include_directories(project_lbfgspp SYSTEM INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/lbfgspp/include")
