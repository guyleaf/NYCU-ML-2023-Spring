add_library(thirdparty INTERFACE)

find_package(Boost 1.75 REQUIRED COMPONENTS filesystem)
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

find_package(unofficial-libsvm CONFIG REQUIRED)

target_link_libraries(thirdparty INTERFACE Boost::filesystem)
target_link_libraries(thirdparty INTERFACE unofficial::libsvm::libsvm)
