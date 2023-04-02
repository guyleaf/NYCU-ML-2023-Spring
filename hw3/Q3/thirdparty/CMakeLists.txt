# Disable GLFW options
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_library(thirdparty INTERFACE)

add_subdirectory(imgui)
add_subdirectory(implot)

target_link_libraries(thirdparty INTERFACE imgui implot)
