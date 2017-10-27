cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

################################################################################
#                               SPQR Example                                   #
################################################################################

set(SUITE_SPARSE_FLOAT /usr/local/google/home/mpoulter/libraries/SuiteSparse/)

link_directories(${SUITE_SPARSE_FLOAT}/lib)
add_executable(spqr_user_guide_example src/spqr_user_guide_example.cc)
target_include_directories(spqr_user_guide_example PUBLIC
                           ${SUITE_SPARSE_FLOAT}/include)
target_link_libraries(spqr_user_guide_example cholmod
                                              glog
                                              spqr
                                              suitesparseconfig)

################################################################################
#                               For Floats                                     #
################################################################################

add_library(sparse_system_float src/sparse_system_float.cc)
target_include_directories(sparse_system_float PUBLIC
                           ${SUITE_SPARSE_FLOAT}/include
                           ${EIGEN3_INCLUDE_DIR})
target_link_libraries(sparse_system_float cholmod spqr)

add_executable(compare_qr_float src/compare_qr_float.cc)
target_link_libraries(compare_qr_float glog
                                       sparse_system_float
                                       suitesparseconfig)

