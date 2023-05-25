#include <iostream>
#include <Magick++.h>

#define NUM_THREADS 12

#include <Eigen/Core>
#include <Eigen/Dense>

#include <optim.hpp>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include <boost/filesystem.hpp>
#include <omp.h>

namespace fs = boost::filesystem;

const std::string IMAGE_FILES[] = {"image1.png", "image2.png"};

int main(int argc, char *argv[])
{

    return 0;
}