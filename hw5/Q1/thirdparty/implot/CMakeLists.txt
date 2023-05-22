cmake_minimum_required(VERSION 3.16...3.25)

# Assuming the canonical version is listed in a single line
# This would be in several parts if picking up from MAJOR, MINOR, etc.
set(version_regex "#define IMPLOT_VERSION[ \t]+\"(.+)\"")

# Read in the line containing the version
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/implot.h"
    version_string REGEX ${version_regex})

# Pick out just the version
string(REGEX REPLACE ${version_regex} "\\1" version_string "${version_string}")

project(implot VERSION ${version_string} LANGUAGES CXX)

add_library(implot implot.cpp implot_items.cpp implot_demo.cpp)
target_include_directories(implot PUBLIC "${PROJECT_SOURCE_DIR}/include")

# All users of this library will need at least C++11
target_compile_features(implot PUBLIC cxx_std_11)

target_link_libraries(implot PUBLIC imgui)
