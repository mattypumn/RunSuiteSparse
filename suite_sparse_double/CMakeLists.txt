cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

################################################################################
#                               For Doubles                                    #
################################################################################

set(SUITE_SPARSE_DOUBLE /usr/local/google/home/mpoulter/Downloads/SuiteSparse/)

link_directories(${SUITE_SPARSE_DOUBLE}/lib)
add_library(sparse_system_double src/sparse_system_double.cc)
target_include_directories(sparse_system_double PUBLIC
                           ${SUITE_SPARSE_DOUBLE}/include)
target_link_libraries(sparse_system_double cholmod spqr)

add_executable(compare_qr_double src/compare_qr_double.cc)
target_link_libraries(compare_qr_double glog sparse_system_double)

