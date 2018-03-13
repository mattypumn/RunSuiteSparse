cmake_minimum_required(VERSION 2.6)

project(RunSuiteSparse)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

################################################################################
#                               SPQR Example                                   #
################################################################################

set(SUITE_SPARSE_FLOAT /home/matt/Libraries/SuiteSparseFloat/)

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
target_link_libraries(sparse_system_float cholmod spqr glog)

add_executable(compare_qr_float src/compare_qr_float.cc)
target_link_libraries(compare_qr_float glog
                                       sparse_system_float
                                       suitesparseconfig)

add_executable(test_spqr_float src/test_spqr_float.cc)
target_link_libraries(test_spqr_float fs_utils
                                      glog
                                      sparse_system_float
                                      eigen_helpers)



set(ANDROID_FLAGS "${COMPILATION_FLAGS} -w -ftree-vectorize -fprefetch-loop-arrays -funroll-loops -DNDEBUG")
set(ANDROID_SPEED "${COMMON_OPTIMIZATION_FLAGS} ${ARCH_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILATION_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS}")


add_library(test_android_float src/test_android_float)

target_link_libraries(test_android_float fs_utils
                                         miniglog
                                         sparse_system_float
                                         eigen_helpers)
include_directories(/home/matt/MarsFramework/MARSFramework/3rd-party/build_android/include/)
