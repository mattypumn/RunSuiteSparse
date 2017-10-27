cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

add_library(eigen_helpers src/eigen_io.cpp)
target_include_directories(eigen_helpers PUBLIC glog
                                                ${EIGEN3_INCLUDE_DIR})
