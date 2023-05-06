add_library(generator_library STATIC generator.cpp)
target_include_directories(generator_library PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(generator_library PUBLIC cxx_std_14)


