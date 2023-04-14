cmake_minimum_required(VERSION 3.16...3.25)

project(mlLib VERSION 2.0
    DESCRIPTION "A library for ML homework"
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    file(GLOB headers_list CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/**/*.h"
        "${PROJECT_SOURCE_DIR}/include/**/*.hpp"
    )

    # IDEs should put the headers in a nice place
    source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${headers_list}
    )

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

add_subdirectory(src)

add_library(mlLib INTERFACE)
target_link_libraries(mlLib INTERFACE algebra_library generator_library)
