# internal library
add_library(imgui_backends_settings INTERFACE)
target_include_directories(imgui_backends_settings INTERFACE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/include/backends"
)

add_library(imgui_backend_glfw imgui_impl_glfw.cpp)
add_library(imgui_backend_opengl3 imgui_impl_opengl3.cpp)

target_link_libraries(imgui_backend_glfw PRIVATE glfw)
target_link_libraries(imgui_backend_opengl3 PRIVATE OpenGL::GL)

target_link_libraries(imgui_backend_glfw PRIVATE imgui_backends_settings)
target_link_libraries(imgui_backend_opengl3 PRIVATE imgui_backends_settings)

# Combinations of backends
add_library(imgui_backend_glfw_opengl3 INTERFACE)
target_link_libraries(imgui_backend_glfw_opengl3 INTERFACE imgui_backend_glfw INTERFACE imgui_backend_opengl3)
