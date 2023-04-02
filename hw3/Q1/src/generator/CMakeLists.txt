add_library(generator_library generator.cpp)
target_include_directories(generator_library PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(generator_library PUBLIC cxx_std_11)
