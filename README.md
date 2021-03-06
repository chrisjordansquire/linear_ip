linear_ip Library
================

This is a straight-forward, no frills implementation of the interior point
linear programming algorithm described in Section 14.2 of "Numerical 
Optimization" by Nocedal and Wright. 

NOTE: This code is not stable. This is currently a toy project, and both
API and ABI breakages will occur as I tweak it. 

Downloading
===========
You can download this library from its github repository. 

    $git clone https://github.com/chrisjordansquire/linear_ip.git

Requirements
============
linear_ip uses Eigen as its matrix library, CMake to create build scripts, and Google Test for its test suite. You can download find more information about
these packages from their homepages at, respectively, www.cmake.org, 
eigen.tuxfamily.org, and code.google.com/p/googletest/. 


Building
========

linear_ip uses CMake. The recommended build process is

    $ mkdir build
    $ cd build
    $ cmake .. -DGTEST_ROOT=${GTEST_ROOT} -DEIGEN_ROOT=${EIGEN_ROOT}

where GTEST_ROOT and EIGEN_ROOT are the local source directories of Eigen 3
and the google testing library.

Then the test suite can be built and run

    $make test_lp_impl
    $make test_lp
    $cd test
    $./test_lp_impl
    $./test_lp

or an example can be built and run
    
    $make example
    $./example/example

The documentation can be built with Doxygen using 'make docs'. The html files are placed in the doc subdirectory of the build directory. 

