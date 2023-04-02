add_library(algebra_library algebra.cpp matrix.cpp)

target_include_directories(algebra_library PUBLIC "${PROJECT_SOURCE_DIR}/include")

# All users of this library will need at least C++11
target_compile_features(algebra_library PUBLIC cxx_std_14)
