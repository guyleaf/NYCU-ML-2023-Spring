#include <string>
#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>

#define NUM_THREADS 12
#define EIGEN_USE_MKL_ALL

#include <Eigen/Core>
#include <Eigen/Dense>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#include <boost/filesystem.hpp>
#include <omp.h>

namespace fs = boost::filesystem;

const std::string DATA_FILE = "input.data";

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-7;

#pragma region Data Structures

using MatrixXd = Eigen::MatrixXd;
using MatrixX2d = Eigen::MatrixX2d;

using VectorXd = Eigen::VectorXd;
using Vector3d = Eigen::Vector3d;

struct LossArguments
{
    const MatrixX2d &data;
    double beta;
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
    // we need to use row-major matrix first, then convert it to column-major order
    return Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>::Map(points.data(), points.size() / 2, 2);
}

#pragma endregion

#pragma region Kernel Function

template <typename DerivedA, typename DerivedB, typename Out = Eigen::Matrix<typename DerivedB::Scalar, DerivedA::RowsAtCompileTime, DerivedA::ColsAtCompileTime>>
Out calculateRationalQuadraticKernel(const Eigen::MatrixBase<DerivedA> &diff, const Eigen::MatrixBase<DerivedB> &kernelParameters)
{
    return (kernelParameters[0] *
            (1 + diff.array() / (2 * kernelParameters[1] * autodiff::detail::pow(kernelParameters[2], 2))).pow(-kernelParameters[1]));
}

template <typename DerivedA, typename DerivedB, typename Out = Eigen::Matrix<typename DerivedB::Scalar, Eigen::Dynamic, Eigen::Dynamic>>
Out calculateRationalQuadraticKernel(const Eigen::MatrixBase<DerivedA> &lhs, const Eigen::MatrixBase<DerivedA> &rhs, const Eigen::MatrixBase<DerivedB> &kernelParameters)
{
    Eigen::Matrix<typename DerivedA::Scalar, Eigen::Dynamic, Eigen::Dynamic> diff = (lhs.replicate(1, lhs.rows()).rowwise() - rhs.transpose()).array().pow(2);
    return calculateRationalQuadraticKernel(diff, kernelParameters);
}

double calculateRationalQuadraticKernel(double lhs, double rhs, const Vector3d &kernelParameters)
{
    VectorXd diff = VectorXd::Constant(1, std::pow(lhs - rhs, 2));
    return calculateRationalQuadraticKernel(diff, kernelParameters).value();
}

VectorXd calculateRationalQuadraticKernel(const VectorXd &lhs, double rhs, const Vector3d &kernelParameters)
{
    VectorXd diff = (lhs.array() - rhs).pow(2);
    return calculateRationalQuadraticKernel(diff, kernelParameters);
}

template<typename Derived, typename Out = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>>
Out calculateCovariance(const MatrixX2d &data, double beta, const Eigen::MatrixBase<Derived> &kernelParameters)
{
    Out covariance = calculateRationalQuadraticKernel(data.col(0), data.col(0), kernelParameters);
    covariance.diagonal().array() += (1 / beta);
    return covariance;
}

#pragma endregion

#pragma region Optimizations

autodiff::real calculateLoss(const autodiff::Vector3real &parameters, const MatrixX2d &data, double beta)
{
    autodiff::MatrixXreal covariance = calculateCovariance(data, beta, parameters);
    return 0.5 * autodiff::detail::log(covariance.determinant())
        + 0.5 * (data.col(1).transpose() * covariance.inverse() * data.col(1)).value()
        + 0.5 * static_cast<double>(data.rows()) * autodiff::detail::log(2 * M_PI);
}

double evaluateOptimFn(const VectorXd &parameters, VectorXd *grad_out, void *args)
{
    auto lossArguments = reinterpret_cast<LossArguments*>(args);
    autodiff::real u;
    autodiff::Vector3real paramtersd = parameters.eval();

    if (grad_out != nullptr)
    {
        auto lossFn = [lossArguments](const autodiff::Vector3real &_paramtersd)
        {
            return calculateLoss(_paramtersd, lossArguments->data, lossArguments->beta);
        };

        *grad_out = autodiff::gradient(lossFn, autodiff::wrt(paramtersd), autodiff::at(paramtersd), u);
    }
    else
    {
        u = calculateLoss(paramtersd, lossArguments->data, lossArguments->beta);
    }

    return u.val();
}

#pragma endregion

#pragma region Custom Functions

void drawPlot(const MatrixX2d &data, const MatrixXd &covariance, double beta, const Vector3d &kernelParameters)
{
}

void modelData(const MatrixX2d &data, double beta)
{
    // variance, alpha, length scale
    Vector3d kernelParameters = Vector3d::Constant(1);

    MatrixXd covariance = calculateCovariance(data, beta, kernelParameters);

    std::cout << covariance << std::endl;

    LossArguments args(data, beta);
    VectorXd optimizedKernelParameters = kernelParameters;
    
    optim::algo_settings_t settings;
    // settings.gd_settings.par_step_size = 1e-4;
    settings.conv_failure_switch = 1;
    settings.vals_bound = true;
    settings.upper_bounds = Vector3d::Constant(1e5);
    settings.lower_bounds = Vector3d::Constant(1e-5);
    settings.print_level = 1;

    optim::bfgs(optimizedKernelParameters, evaluateOptimFn, reinterpret_cast<void*>(&args), settings);

    MatrixXd optimizedCovariance = calculateCovariance(data, beta, optimizedKernelParameters);

    std::cout << optimizedCovariance << std::endl << std::endl;
    std::cout << optimizedKernelParameters << std::endl;
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
