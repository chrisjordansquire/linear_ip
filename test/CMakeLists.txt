cmake_minimum_required(VERSION 2.8.5)

set(gtest_SRC
    ${GTEST_ROOT}/src/gtest-all.cc
    ${GTEST_ROOT}/src/gtest_main.cc
    )

set(test_lp_SRC
    test_lp.cpp
    ${gtest_SRC})

set(test_lp_impl_SRC
    test_lp_impl.cpp
    ${gtest_SRC})


include_directories(
    ${GTEST_ROOT}/include 
    ${GTEST_ROOT})

add_executable(test_lp ${test_lp_SRC})
target_link_libraries(test_lp lp lp_impl pthread)

add_executable(test_lp_impl ${test_lp_impl_SRC})
target_link_libraries(test_lp_impl lp_impl pthread)
