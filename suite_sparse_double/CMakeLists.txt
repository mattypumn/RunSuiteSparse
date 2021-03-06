cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-O3")

################################################################################
#                               For Doubles                                    #
################################################################################

set(SUITE_SPARSE_DOUBLE /home/matt/Libraries/SuiteSparseDouble/)

## For tbb.
# set(SUITE_SPARSE_DOUBLE /usr/local/google/home/mpoulter/Downloads/SuiteSparse_Tbb/)

link_directories(${SUITE_SPARSE_DOUBLE}/lib)
add_library(sparse_system_double src/sparse_system_double.cc)
target_include_directories(sparse_system_double PUBLIC
                           ${SUITE_SPARSE_DOUBLE}/include
                           ${EIGEN3_INCLUDE_DIR})
target_link_libraries(sparse_system_double cholmod spqr)

add_executable(compare_qr_double src/compare_qr_double.cc)
target_link_libraries(compare_qr_double glog
                                        sparse_system_double
                                        suitesparseconfig
                                        eigen_helpers)

add_executable(compare_split_qr src/compare_split_qr.cc)
target_link_libraries(compare_split_qr glog
                                       sparse_system_double
                                       eigen_helpers)

add_executable(test_spqr_double src/test_spqr_double.cc)
target_link_libraries(test_spqr_double fs_utils
                                       glog
                                       sparse_system_double
                                       eigen_helpers)
