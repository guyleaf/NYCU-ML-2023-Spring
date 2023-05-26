add_library(cluster STATIC kmeans.cpp spectral.cpp)
target_include_directories(cluster PUBLIC "${PROJECT_SOURCE_DIR}/include")
# All users of this library will need at least C++11
target_compile_features(cluster PUBLIC cxx_std_20)