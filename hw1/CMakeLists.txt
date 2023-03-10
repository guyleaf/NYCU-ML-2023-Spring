cmake_minimum_required(VERSION 3.16...3.25)

project(mlhw1 VERSION 1.0
    DESCRIPTION "Machine Learning HW1"
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app
    # Note this needs to be done in the main CMakeLists
    # since it calls enable_testing, which must be in the
    # main CMakeLists.
    # include(CTest)

    # Docs only available if this is the main app
    # find_package(Doxygen)
    # if(Doxygen_FOUND)
    # add_subdirectory(docs)
    # else()
    # message(STATUS "Doxygen not found, not building docs")
    # endif()
endif()

find_package(OpenGL REQUIRED)

add_subdirectory(src)
add_subdirectory(thirdparty)

# Main executable code
add_executable(mlhw1 main.cpp)
target_compile_features(mlhw1 PRIVATE cxx_std_11)

target_link_libraries(mlhw1 PRIVATE OpenGL::GL glfw imgui implot algebra_library)

add_custom_command(
    TARGET mlhw1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/samples"
    "${CMAKE_CURRENT_BINARY_DIR}/samples")
