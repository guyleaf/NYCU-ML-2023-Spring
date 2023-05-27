#include <iostream>
#include <vector>
#include <string>
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

enum InitMethod
{
    Random,
    Kmeansplusplus,
};

#pragma region Data processing

std::pair<Eigen::MatrixX3d, Eigen::MatrixX2i> preprocess(Magick::Image &image)
{
    auto rows = image.rows();
    auto columns = image.columns();
    auto size = rows * columns;

    std::vector<unsigned char> tmp(size * 3);
    image.writePixels(Magick::QuantumType::RGBQuantum, tmp.data());
    std::vector<double> pixels(size * 3);
    std::move(tmp.begin(), tmp.end(), pixels.begin());

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

    using MatrixX3dRowMajor = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatrixX2iRowMajor = Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>;

    return std::make_pair<Eigen::MatrixX3d, Eigen::MatrixX2i>(
        MatrixX3dRowMajor::Map(pixels.data(), size, 3), MatrixX2iRowMajor::Map(coordinates.data(), size, 2));
}

#pragma endregion

Eigen::MatrixXd calculateKernel(const Eigen::MatrixX3d &pixels, const Eigen::MatrixX2i &coordinates, double gamma1, double gamma2)
{
    auto colorKernel = mlhw6::rbf(pixels, pixels, gamma1);

    const Eigen::MatrixX2d &tmp = coordinates.cast<double>();
    auto coordinateKernel = mlhw6::rbf(tmp, tmp, gamma2);

    return colorKernel.cwiseProduct(coordinateKernel);
}

void run(const fs::path &path, int k, InitMethod init, double gamma1, double gamma2)
{
    Magick::InitializeMagick(nullptr);
    Magick::Image image;

    for (auto imageFile : IMAGE_FILES)
    {
        // read image
        image.read((path / imageFile).generic_string());

        // extract RGB values and coordinates
        auto [pixels, coordinates] = preprocess(image);

        // calculate kernel
        auto kernel = calculateKernel(pixels, coordinates, gamma1, gamma2);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] << " <data path> <number of cluster> <init> <gamma1> <gamma2>" << std::endl;
        return 1;
    }

    argv++;

    fs::path path(*argv);
    if (!fs::is_directory(path))
    {
        std::cerr << "The data path is not a directory or not existed." << std::endl;
        return 1;
    }

    auto k = std::stoi(*argv);
    if (!(k > 0))
    {
        std::cerr << "The number of cluster should be larger than 0." << std::endl;
        return 1;
    }

    argv++;

    auto init = static_cast<InitMethod>(std::stoi(*argv));

    argv++;

    auto gamma1 = std::stod(*argv);

    argv++;

    auto gamma2 = std::stod(*argv);

    run(path, k, init, gamma1, gamma2);

    return 0;
}