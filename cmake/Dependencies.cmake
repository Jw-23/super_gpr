include("${CMAKE_CURRENT_LIST_DIR}/../CPM.cmake")

if(CMAKE_CXX_STANDARD LESS 17)
  # These are the newest release lines that build with a C++14-only compiler.
  set(PROJECT_GOOGLETEST_VERSION 1.16.0)
  set(PROJECT_BENCHMARK_VERSION 1.9.0)
else()
  set(PROJECT_GOOGLETEST_VERSION 1.18.0)
  set(PROJECT_BENCHMARK_VERSION 1.9.5)
endif()

set(PROJECT_SPDLOG_VERSION 1.17.0)
set(PROJECT_FMT_VERSION 12.1.0)
set(PROJECT_EIGEN_VERSION 3.4.0)

message(
  STATUS
  "Dependency compatibility: C++${CMAKE_CXX_STANDARD}, "
  "GoogleTest ${PROJECT_GOOGLETEST_VERSION}, "
  "Benchmark ${PROJECT_BENCHMARK_VERSION}, "
  "spdlog ${PROJECT_SPDLOG_VERSION}, "
  "fmt ${PROJECT_FMT_VERSION}, "
  "Eigen ${PROJECT_EIGEN_VERSION}"
)

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

if(PROJECT_ENABLE_FMT)
  CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    VERSION ${PROJECT_FMT_VERSION}
    GIT_TAG ${PROJECT_FMT_VERSION}
    OPTIONS "FMT_TEST OFF" "FMT_DOC OFF" "FMT_INSTALL OFF"
  )
endif()

if(PROJECT_ENABLE_SPDLOG)
  if(PROJECT_ENABLE_FMT)
    set(PROJECT_SPDLOG_FMT_OPTION "SPDLOG_FMT_EXTERNAL ON")
  else()
    set(PROJECT_SPDLOG_FMT_OPTION "SPDLOG_FMT_EXTERNAL OFF")
  endif()

  CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION ${PROJECT_SPDLOG_VERSION}
    OPTIONS
      "SPDLOG_BUILD_EXAMPLE OFF"
      "SPDLOG_BUILD_TESTS OFF"
      "SPDLOG_INSTALL OFF"
      "${PROJECT_SPDLOG_FMT_OPTION}"
  )
endif()

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  CPMAddPackage(
    NAME eigen
    URL https://github.com/eigen-mirror/eigen/archive/refs/tags/3.4.0.tar.gz
    DOWNLOAD_ONLY YES
  )
  add_library(project_eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS project_eigen)
  target_include_directories(project_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
endif()

CPMAddPackage(
  NAME lbfgspp
  URL https://github.com/yixuan/LBFGSpp/archive/c524a407fb85b74807f53de5a3ca2ddbcc164e54.tar.gz
  DOWNLOAD_ONLY YES
)
add_library(project_lbfgspp INTERFACE)
target_include_directories(project_lbfgspp SYSTEM INTERFACE "${lbfgspp_SOURCE_DIR}/include")
