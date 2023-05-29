#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <utility>
#include <random>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <matplot/matplot.h>

#include <boost/filesystem.hpp>
#include <omp.h>

#include <mlhw6/kernel.hpp>
#include <mlhw6/cluster.h>

namespace fs = boost::filesystem;

const std::string IMAGE_FILES[] = {"image1.png", "image2.png"};

enum SpectralClusteringType
{
    Unnormalized,
    Normalized
};

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

Eigen::MatrixXd calculateKernel(const Eigen::MatrixX3d &pixels, const Eigen::MatrixX2i &coordinates, double gamma1, double gamma2)
{
    auto colorKernel = mlhw6::rbf(pixels, pixels, gamma1);

    const Eigen::MatrixX2d &tmp = coordinates.cast<double>();
    auto coordinateKernel = mlhw6::rbf(tmp, tmp, gamma2);

    return colorKernel.cwiseProduct(coordinateKernel);
}

cv::Mat drawMask(const Eigen::VectorXi &labels, int numberOfClusters, unsigned int width, unsigned int height)
{
    Eigen::VectorX<unsigned char> maskData = labels.cast<unsigned char>();
    cv::Mat mask = cv::Mat(cv::Size(width, height), CV_8UC1, reinterpret_cast<void *>(maskData.data()));

    mask *= (255 / numberOfClusters);

    cv::Mat bgrMask;
    cv::cvtColor(mask, bgrMask, cv::COLOR_GRAY2BGR);
    cv::applyColorMap(bgrMask, bgrMask, cv::COLORMAP_COOL);
    return bgrMask;
}

void plotEigenSpace(const std::string &path, const Eigen::MatrixXd &eigenMatrix)
{
    matplot::figure();

    const Eigen::VectorXd &vecX = eigenMatrix.col(0);
    const Eigen::VectorXd &vecY = eigenMatrix.col(1);

    std::vector<double> x(vecX.begin(), vecX.end());
    std::vector<double> y(vecY.begin(), vecY.end());
    matplot::scatter(x, y);
    matplot::save(path);
}

void run(const fs::path &path, mlhw6::BaseSpectralClustering &model, int numberOfClusters, double gamma1, double gamma2)
{
    int fps = 30;
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer;

    for (auto imageFile : IMAGE_FILES)
    {
        std::cout << imageFile << std::endl;

        auto fileName = (path / imageFile).generic_string();

        // read image
        auto image = cv::imread(fileName, cv::ImreadModes::IMREAD_COLOR);

        // extract RGB values and coordinates
        auto [pixels, coordinates] = preprocess(image);

        // calculate kernel
        auto kernel = calculateKernel(pixels, coordinates, gamma1, gamma2);

        model.fit(kernel);
        const std::vector<Eigen::VectorXi> &fittingHistory = model.getFittingHistory();

        plotEigenSpace(imageFile + "_eigen.jpg", model.getEigenMatrix());

        writer.open(imageFile + "_video.mp4", codec, fps, image.size());
        // check if we succeeded
        if (!writer.isOpened())
        {
            throw std::runtime_error("Could not open the output video file for write");
        }

        for (int i = 0; i < fps; i++)
        {
            writer.write(image);
        }

        cv::Mat result;
        cv::Mat mask;
        for (std::size_t i = 1; i < fittingHistory.size(); i++)
        {
            mask = drawMask(fittingHistory[i], numberOfClusters, image.cols, image.rows);
            cv::addWeighted(image, 0.5, mask, 0.5, 0, result);
            for (int i = 0; i < fps; i++)
            {
                writer.write(result);
            }
        }

        writer.release();

        cv::imwrite(imageFile + "_mask.png", mask);
        cv::imwrite(imageFile + "_final.png", result);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] << " <data path> <number of cluster> <spectral clustering type> <init> <gamma1> <gamma2>" << std::endl;
        return 1;
    }

    argv++;

    fs::path path(*argv);
    if (!fs::is_directory(path))
    {
        std::cerr << "The data path is not a directory or not existed." << std::endl;
        return 1;
    }

    argv++;

    auto numberOfClusters = std::stoi(*argv);
    if (!numberOfClusters > 0)
    {
        std::cerr << "The number of cluster should be larger than 0." << std::endl;
        return 1;
    }

    argv++;

    auto type = static_cast<SpectralClusteringType>(std::stoi(*argv));

    argv++;

    auto init = static_cast<mlhw6::KMeansInitMethods>(std::stoi(*argv));

    argv++;

    auto gamma1 = std::stod(*argv);

    argv++;

    auto gamma2 = std::stod(*argv);

    switch (type)
    {
    case SpectralClusteringType::Unnormalized:
    {
        auto model = mlhw6::SpectralClustering(numberOfClusters, init);
        run(path, model, numberOfClusters, gamma1, gamma2);
        break;
    }
    case SpectralClusteringType::Normalized:
    {
        auto model = mlhw6::NormalizedSpectralClustering(numberOfClusters, init);
        run(path, model, numberOfClusters, gamma1, gamma2);
        break;
    }
    default:
        std::cerr << "Unknown spectral clustering type." << std::endl;
        return 1;
    }

    return 0;
}