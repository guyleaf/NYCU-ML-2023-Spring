add_library(algebra_library algebra.cpp matrix.cpp)

target_include_directories(algebra_library PUBLIC "${PROJECT_SOURCE_DIR}/include")

# All users of this library will need at least C++11
target_compile_features(algebra_library PUBLIC cxx_std_11)

file(GLOB headers_list CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/**/*.h"
    "${PROJECT_SOURCE_DIR}/include/**/*.hpp"
)

# IDEs should put the headers in a nice place
source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${headers_list})
