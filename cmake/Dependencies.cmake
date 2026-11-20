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

if(NOT TARGET LAPACKE::LAPACKE)
  find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h REQUIRED)
  find_library(LAPACKE_LIBRARY NAMES lapacke REQUIRED)
  find_package(LAPACK REQUIRED)
  add_library(LAPACKE::LAPACKE UNKNOWN IMPORTED)
  set_target_properties(LAPACKE::LAPACKE PROPERTIES
    IMPORTED_LOCATION "${LAPACKE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES LAPACK::LAPACK
  )
endif()

add_library(project_lbfgspp INTERFACE)
target_include_directories(project_lbfgspp SYSTEM INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/../third_party/lbfgspp/include")

find_package(Boost 1.83 QUIET CONFIG)
if(TARGET Boost::headers)
  set(PROJECT_OUTCOME_HEADERS Boost::headers)
else()
  CPMAddPackage(
    NAME boost_headers
    URL https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-b2-nodocs.tar.xz
    URL_HASH SHA256=ad9ce2c91bc0977a7adc92d51558f3b9c53596bb88246a280175ebb475da1762
    DOWNLOAD_ONLY YES
  )
  add_library(project_boost_headers INTERFACE)
  target_include_directories(project_boost_headers SYSTEM INTERFACE "${boost_headers_SOURCE_DIR}")
  set(PROJECT_OUTCOME_HEADERS project_boost_headers)
endif()
