cmake_minimum_required(VERSION 3.16...3.25)
project(optim VERSION 1.0 LANGUAGES CXX)

add_library(optim INTERFACE)
target_include_directories(optim INTERFACE "${PROJECT_SOURCE_DIR}/include")

# All users of this library will need at least C++11
target_compile_features(optim INTERFACE cxx_std_14)
