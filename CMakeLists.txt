cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)


#
# add_executable(spqr_user_guide_example src/spqr_user_guide_example.cc)
# target_include_directories(spqr_user_guide_example PUBLIC
#                            ${SUITE_SPARSE_FLOAT}/include)
# link_directories(${SUITE_SPARSE_FLOAT}/lib)
# target_link_libraries(spqr_user_guide_example cholmod spqr spqrf glog)



################################################################################
#                               For Floats                                     #
################################################################################

set(SUITE_SPARSE_FLOAT /usr/local/google/home/mpoulter/libraries/SuiteSparse/)

link_directories(${SUITE_SPARSE_FLOAT}/lib)
add_library(sparse_qr_float src/sparse_qr_float.cc)
target_include_directories(sparse_qr_float PUBLIC ${SUITE_SPARSE_FLOAT}/include)
target_link_libraries(sparse_qr_float cholmod spqr)

add_executable(compare_qr_float src/compare_qr_float.cc)
target_link_libraries(compare_qr_float glog sparse_qr_float)

#
#
# ################################################################################
#                               For Doubles                                    #
################################################################################
# set(SUITE_SPARSE_DOUBLE /usr/local/google/home/mpoulter/Downloads/SuiteSparse/)
#
# link_directories(${SUITE_SPARSE_DOUBLE}/lib)
# add_library(sparse_qr_double src/sparse_qr_double.cc)
# target_include_directories(sparse_qr_double PUBLIC
#                            ${SUITE_SPARSE_DOUBLE}/include)
# target_link_libraries(sparse_qr_double cholmod spqr)
#
# add_executable(compare_qr_double src/compare_qr_double.cc)
# target_link_libraries(compare_qr_double glog sparse_qr_double)

