# Disable GLFW options
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Eigen options
set(CMAKE_CXX_FLAGS "-fopenmp -march=native -ffp-contract=fast" CACHE STRING "" FORCE)

# autodiff options
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)

# MKL options
set(MKL_INTERFACE "lp64")

# LAPACK options
# set(BLA_VENDOR "Intel10_64lp")

add_subdirectory(optim)
add_subdirectory(imgui)
add_subdirectory(implot)
add_subdirectory(autodiff)

add_library(thirdparty INTERFACE)

find_package(MKL CONFIG REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(Boost 1.75 REQUIRED COMPONENTS filesystem)
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

target_link_libraries(thirdparty INTERFACE LAPACK::LAPACK)
target_compile_options(thirdparty INTERFACE "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>")
target_include_directories(thirdparty INTERFACE "$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>")
target_link_libraries(thirdparty INTERFACE "$<LINK_ONLY:MKL::MKL>")
target_link_libraries(thirdparty INTERFACE Eigen3::Eigen Boost::filesystem optim autodiff::autodiff)
target_link_libraries(thirdparty INTERFACE imgui implot)
