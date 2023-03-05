cmake_minimum_required(VERSION 3.16...3.25)

# Assuming the canonical version is listed in a single line
# This would be in several parts if picking up from MAJOR, MINOR, etc.
set(version_regex "#define IMGUI_VERSION[ \t]+\"(.+)\"")

# Read in the line containing the version
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/imgui.h"
    version_string REGEX ${version_regex})

# Pick out just the version
string(REGEX REPLACE ${version_regex} "\\1" version_string "${version_string}")

project(imgui VERSION ${version_string} LANGUAGES CXX)

add_library(imgui imgui.cpp imgui_widgets.cpp imgui_tables.cpp imgui_draw.cpp imgui_demo.cpp)
target_include_directories(imgui PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_subdirectory(backends)

# All users of this library will need at least C++11
target_compile_features(imgui PUBLIC cxx_std_11)

target_link_libraries(imgui INTERFACE imgui_backend_glfw_opengl3)
