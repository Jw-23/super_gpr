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

CPMAddPackage(
  NAME lbfgspp
  URL https://github.com/yixuan/LBFGSpp/archive/c524a407fb85b74807f53de5a3ca2ddbcc164e54.tar.gz
  URL_HASH SHA256=8fbb7037c4e0bf0d83ced74463092154038bc0bbb43f77fe38f2bc7c3414c2d8
  DOWNLOAD_ONLY YES
)
add_library(project_lbfgspp INTERFACE)
target_include_directories(project_lbfgspp SYSTEM INTERFACE "${lbfgspp_SOURCE_DIR}/include")
