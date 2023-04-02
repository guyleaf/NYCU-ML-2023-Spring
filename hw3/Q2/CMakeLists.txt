cmake_minimum_required(VERSION 3.16...3.25)

project(mlhw3_q2 VERSION 1.0
    DESCRIPTION "Machine Learning HW3 Q2"
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

add_subdirectory(libs)

# Main executable code
add_executable(mlhw3_q2 main.cpp)

target_compile_features(mlhw3_q2 PRIVATE cxx_std_14)

target_include_directories(mlhw3_q2 PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(mlhw3_q2 PRIVATE libs)
