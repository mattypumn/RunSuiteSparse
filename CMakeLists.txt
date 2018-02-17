cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Release)

set(EIGEN3_INCLUDE_DIR /home/matt/MarsFramework/MARSFramework/3rd-party/eigen3)

find_package(glog 0.3.4)

add_subdirectory(suite_sparse_float)
add_subdirectory(suite_sparse_double)
add_subdirectory(eigen_helpers)
add_subdirectory(fs_utils)

include_directories(eigen_helpers/include)
include_directories(suite_sparse_double/include)
include_directories(suite_sparse_float/include)
include_directories(fs_utils/include)
