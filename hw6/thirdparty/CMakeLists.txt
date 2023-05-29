add_library(thirdparty INTERFACE)

find_package(OpenBLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.75 REQUIRED COMPONENTS filesystem)
find_package(OpenCV CONFIG REQUIRED)
find_package(Matplot++ REQUIRED)

target_compile_definitions(thirdparty INTERFACE ${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
target_compile_definitions(thirdparty INTERFACE EIGEN_USE_BLAS)

target_include_directories(thirdparty INTERFACE ${OpenCV_INCLUDE_DIRS})
target_include_directories(thirdparty INTERFACE ${Boost_INCLUDE_DIRS})

target_link_libraries(thirdparty INTERFACE OpenBLAS::OpenBLAS LAPACK::LAPACK)
target_link_libraries(thirdparty INTERFACE Eigen3::Eigen)
target_link_libraries(thirdparty INTERFACE Boost::filesystem)
target_link_libraries(thirdparty INTERFACE ${OpenCV_LIBS})
target_link_libraries(thirdparty INTERFACE Matplot++::matplot)
