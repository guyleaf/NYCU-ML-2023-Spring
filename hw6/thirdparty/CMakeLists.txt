add_library(thirdparty INTERFACE)

find_package(LAPACK REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.75 REQUIRED COMPONENTS filesystem)
find_package(OpenCV CONFIG REQUIRED)

target_compile_definitions(thirdparty INTERFACE ${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
target_compile_definitions(thirdparty INTERFACE EIGEN_USE_BLAS EIGEN_USE_LAPACKE EIGEN_USE_LAPACKE_STRICT)
target_compile_definitions(thirdparty INTERFACE OPTIM_ENABLE_EIGEN_WRAPPERS)

target_include_directories(thirdparty INTERFACE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(thirdparty INTERFACE LAPACK::LAPACK)
target_link_libraries(thirdparty INTERFACE Eigen3::Eigen)
target_link_libraries(thirdparty INTERFACE Boost::filesystem)
target_link_libraries(thirdparty INTERFACE ${OpenCV_LIBS})
