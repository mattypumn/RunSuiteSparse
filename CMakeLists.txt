cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(suite_sparse_float/)
add_subdirectory(suite_sparse_double/)

