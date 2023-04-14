add_library(libs INTERFACE)
target_link_libraries(libs INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/libgenerator_library.a" "${CMAKE_CURRENT_SOURCE_DIR}/libalgebra_library.a")
target_include_directories(libs INTERFACE "${CMAKE_SOURCE_DIR}/include")