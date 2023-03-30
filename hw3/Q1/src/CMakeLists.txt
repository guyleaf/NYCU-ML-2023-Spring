add_subdirectory("algebra")

add_library(generator_library INTERFACE)
target_include_directories(generator_library INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(generator_library INTERFACE cxx_std_11)

add_library(src_library INTERFACE)
target_link_libraries(src_library INTERFACE algebra_library generator_library)

file(GLOB headers_list CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/**/*.h"
    "${PROJECT_SOURCE_DIR}/include/**/*.hpp"
)

# IDEs should put the headers in a nice place
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${headers_list})