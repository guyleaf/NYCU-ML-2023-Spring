add_subdirectory("cluster")

add_library(src INTERFACE)
target_link_libraries(src INTERFACE cluster)
target_include_directories(src INTERFACE "${PROJECT_SOURCE_DIR}/include")
