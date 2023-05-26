#include <iostream>
#include <vector>
#include <utility>

#include <Magick++.h>
#include <Eigen/Dense>
#include <optim.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <boost/filesystem.hpp>
#include <omp.h>

#include <mlhw6/kernel.hpp>

namespace fs = boost::filesystem;

const std::string IMAGE_FILES[] = {"image1.png", "image2.png"};

#pragma region Data preprocessing

std::pair<Eigen::MatrixX3d, Eigen::MatrixX2i> preprocess(Magick::Image &image)
{
    auto rows = image.rows();
    auto columns = image.columns();
    auto size = rows * columns;

    std::vector<unsigned char> tmp(size * 3);
    image.writePixels(Magick::QuantumType::RGBQuantum, tmp.data());
    std::vector<double> pixels(size * 3);
    std::move(tmp.begin(), tmp.end(), pixels.begin());
    

    std::vector<int> points(size * 2);
#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < columns; j++)
        {
            auto index = (i * columns + j) * 2;
            points[index] = i;
            points[index + 1] = j;
        }
    }

    using MatrixX3dRowMajor = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatrixX2iRowMajor = Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>;

    return std::make_pair<Eigen::MatrixX3d, Eigen::MatrixX2d>(MatrixX3dRowMajor::Map(pixels.data(), pixels.size()), MatrixX2iRowMajor::Map(points.data(), points.size()));
}

#pragma endregion

int main(int argc, char *argv[])
{
    Magick::InitializeMagick(nullptr);
    Magick::Image image;

    return 0;
}