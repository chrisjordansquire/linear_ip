cmake_minimum_required(VERSION 2.8.5)

project(linear_ip)

set(linear_ip_VERSION_MAJOR 0)
set(linear_ip_VERSION_MINOR 1)

set(CMAKE_VERBOSE_MAKEFILE 1)

if(NOT CMAKE_BUILD_TYPE)
    message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("\n===================================")
        message("\n-- Build type: Debug. Performance will suffer.")
        message("\n===================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_C_FLAGS_DEBUG "-g3 -O1 -UNDEBUG -Wall -Wno-address")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O1 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

if( NOT EXISTS ${EIGEN_ROOT}/Eigen/Dense )
    message(FATAL_ERROR "EIGEN_ROOT (${EIGEN_ROOT}) is not a valid Eigen package")
endif()

if( NOT EXISTS ${GTEST_ROOT}/src/gtest_main.cc )
    message(FATAL_ERROR "GTEST_ROOT (${GTEST_ROOT}) is not a valid Gtest package.")
endif()

include_directories(include ${EIGEN_ROOT})

set(lp_impl_SRC
    src/lp_impl.cpp
    )

set(lp_SRC
    src/lp.cpp)

add_library(lp_impl SHARED ${lp_impl_SRC})
add_library(lp SHARED ${lp_SRC})
target_link_libraries(lp lp_impl)

option(BUILD_TESTS "Build test suite." ON)
if(${BUILD_TESTS})
    message("-- Build the test suite.")
    add_subdirectory(test)
endif(${BUILD_TESTS})

option(BUILD_DOCUMENTATION "Build doxygen documentation." ON)
if(${BUILD_DOCUMENTATION})
    message("-- Build the doxygen docs.")
    add_subdirectory(doc)
endif(${BUILD_DOCUMENTATION})

option(BUILD_EXAMPLES "Build examples" ON)
if(${BUILD_EXAMPLES})
    message("-- Build the examples")
    add_subdirectory(example)
endif(${BUILD_EXAMPLES})


