cmake_minimum_required(VERSION 2.6)

add_library(fs_utils src/fs_utils.cc)
include_directories(fs_utils ${EIGEN3_INCLUDE_DIR})
