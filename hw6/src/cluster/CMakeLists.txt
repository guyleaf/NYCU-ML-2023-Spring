add_library(cluster STATIC kmeans.cpp spectral.cpp)
target_include_directories(cluster PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(cluster PUBLIC cxx_std_20)

target_link_libraries(cluster PUBLIC thirdparty)