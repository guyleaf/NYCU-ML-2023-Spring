cmake_minimum_required(VERSION 3.16...3.25)

project(mlhw6 VERSION 1.0
    DESCRIPTION "Machine Learning HW4 Q1"
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    set(CMAKE_CXX_STANDARD 20)

    set(CMAKE_CXX_FLAGS "-fopenmp -march=native -ffp-contract=fast" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=address -g")

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_subdirectory(src)
add_subdirectory(thirdparty)

add_library(common INTERFACE)
target_compile_features(common INTERFACE cxx_std_20)
target_include_directories(common INTERFACE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(common INTERFACE src thirdparty)

# Main executable code
add_executable(mlhw6_kernel_kmeans kernel_kmeans.cpp)
target_link_libraries(mlhw6_kernel_kmeans PRIVATE common)

add_executable(mlhw6_spectral_clustering spectral_clustering.cpp)
target_link_libraries(mlhw6_spectral_clustering PRIVATE common)

add_custom_command(
    TARGET mlhw6_kernel_kmeans POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/data"
    "${CMAKE_CURRENT_BINARY_DIR}/data")

add_custom_command(
    TARGET mlhw6_spectral_clustering POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/data"
    "${CMAKE_CURRENT_BINARY_DIR}/data")
