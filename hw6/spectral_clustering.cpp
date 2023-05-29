#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <utility>
#include <random>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <boost/filesystem.hpp>
#include <omp.h>

#include <mlhw6/kernel.hpp>
#include <mlhw6/cluster.h>

namespace fs = boost::filesystem;

const std::string IMAGE_FILES[] = {"image1.png", "image2.png"};

#pragma region Data processing

std::pair<Eigen::MatrixX3d, Eigen::MatrixX2i> preprocess(const cv::Mat &image)
{
    auto rows = image.rows;
    auto columns = image.cols;
    auto size = rows * columns;

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    std::vector<int> coordinates(size * 2);
#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < columns; j++)
        {
            auto index = (i * columns + j) * 2;
            coordinates[index] = i;
            coordinates[index + 1] = j;
        }
    }

    using MatrixX3ucRowMajor = Eigen::Matrix<unsigned char, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatrixX2iRowMajor = Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>;

    return std::make_pair<Eigen::MatrixX3d, Eigen::MatrixX2i>(
        MatrixX3ucRowMajor::Map(rgb.data, size, 3).cast<double>(), MatrixX2iRowMajor::Map(coordinates.data(), size, 2));
}

#pragma endregion

int main(int argc, char *argv[])
{

    return 0;
}