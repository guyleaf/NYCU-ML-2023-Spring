#include <string>
#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#include <cmath>
#include <random>

#define NUM_THREADS 12
#define EIGEN_USE_MKL_ALL

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/filesystem.hpp>
#include <omp.h>

namespace fs = boost::filesystem;

const std::string DATA_FILE = "input.data";

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-7;

#pragma region Data Structures

using MatrixXd = Eigen::MatrixXd;
using MatrixX2d = Eigen::MatrixX2d;

using VectorXd = Eigen::VectorXd;

struct KernelParameters
{
    double variance = 1;
    double alpha = 1;
    double lengthScale = 1;
};

#pragma endregion

#pragma region File Processing

MatrixX2d parseDataFile(const fs::path &path)
{
    std::ifstream ifs(path.generic_string());
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    std::vector<double> points{std::istream_iterator<double>(ifs), std::istream_iterator<double>()};

    ifs.close();

    // because the vector is stored in row-major order
    // we need to use row-major matrix, then convert it to column-major order
    return Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>::Map(points.data(), points.size() / 2, 2);
}

#pragma endregion

#pragma region Custom Functions

void drawPlot(const MatrixX2d &data, const MatrixXd &covariance, double beta, const KernelParameters &kernelParameters)
{
}

double calculateRationalQuadraticKernel(double lhs, double rhs, const KernelParameters &kernelParameters)
{
    return kernelParameters.variance * 
        std::pow(1 + std::pow(lhs - rhs, 2) / (2 * kernelParameters.alpha * std::pow(kernelParameters.lengthScale, 2)), -kernelParameters.alpha);
}

VectorXd calculateRationalQuadraticKernel(VectorXd lhs, double rhs, const KernelParameters &kernelParameters)
{
    return lhs.unaryExpr([rhs, kernelParameters](double x)
                         { return calculateRationalQuadraticKernel(x, rhs, kernelParameters); });
}

MatrixXd calculateCovariance(const MatrixX2d &data, double beta, const KernelParameters &kernelParameters)
{
    auto fn = [data, kernelParameters](Eigen::Index i, Eigen::Index j)
    {
        return calculateRationalQuadraticKernel(data[i], data[j], kernelParameters);
    };

    MatrixXd covariance = MatrixXd::NullaryExpr(data.rows(), data.rows(), fn);
    covariance.diagonal().array() += (1 / beta);
    return covariance;
}

void modelData(const MatrixX2d &data, double beta)
{
    KernelParameters kernelParameters;
    MatrixXd covariance = calculateCovariance(data, beta, kernelParameters);

    KernelParameters optimizedKernelParameters = kernelParameters;
    double dAlpha;
    double dLengthScale;
    do
    {
        dAlpha = optimizedKernelParameters.alpha;
        dLengthScale = optimizedKernelParameters.lengthScale;

        dAlpha = std::abs(dAlpha - optimizedKernelParameters.alpha);
        dLengthScale = std::abs(dLengthScale - optimizedKernelParameters.lengthScale);
    } while (dAlpha >= STOP_APPROXIMATION_THRESHOLD || dLengthScale >= STOP_APPROXIMATION_THRESHOLD);

    MatrixXd optimizedCovariance = calculateCovariance(data, beta, optimizedKernelParameters);
}

#pragma endregion

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <data path> <beta>" << std::endl;
        return 1;
    }

    omp_set_num_threads(NUM_THREADS);
    std::cout << "Running in " << Eigen::nbThreads() << " threads" << std::endl;

    argv++;

    fs::path path(*argv);
    if (!fs::is_directory(path))
    {
        std::cerr << "The data path is not a directory or not existed." << std::endl;
        return 1;
    }

    argv++;

    double beta = std::stod(*argv);

    // parse data file
    auto data = parseDataFile(path / DATA_FILE);

    modelData(data, beta);
    return 0;
}
