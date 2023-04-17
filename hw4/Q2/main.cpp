#include <random>
#include <string>
#include <iostream>
#include <array>
#include <vector>
#include <variant>
#include <iterator>
#include <limits>

#define NUM_THREADS 12
#define EIGEN_USE_MKL_ALL

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/filesystem.hpp>
#include <omp.h>

namespace fs = boost::filesystem;

const std::string TRAIN_IMAGES_FILE = "train-images.idx3-ubyte";
const std::string TRAIN_LABELS_FILE = "train-labels.idx1-ubyte";

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-7;

#pragma region Data Structures

// for MNIST images
constexpr int IMAGE_DATA_ID = 2051;
constexpr int LABEL_DATA_ID = 2049;

using MatrixX10d = Eigen::Matrix<double, Eigen::Dynamic, 10>;
using Matrix10Xd = Eigen::Matrix<double, 10, Eigen::Dynamic>;
using MatrixXb = Eigen::Matrix<std::byte, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Vector10d = Eigen::Vector<double, 10>;
using Vector10i = Eigen::Vector<int, 10>;
using VectorXb = Eigen::VectorX<std::byte>;

using STLArray4b = std::array<std::byte, 4>;

class RandomGenerator
{
public:
    RandomGenerator()
    {
        std::random_device rd;
        std::seed_seq seed{rd()};
        this->rng = std::mt19937_64(seed);
    }

    Eigen::MatrixXd randu(std::size_t n, std::size_t m, double a, double b)
    {
        std::uniform_real_distribution uniform_dist(a, b);
        auto uniform = [&uniform_dist, &rng = this->rng]()
        { return uniform_dist(rng); };
        return Eigen::MatrixXd::NullaryExpr(n, m, uniform);
    }

private:
    std::mt19937_64 rng;
};

struct ImageSize
{
    int height = 0;
    int width = 0;

    int size() const
    {
        return height * width;
    }
};

using PixelsOrLabels = std::variant<MatrixXb, VectorXb>;

struct FileBody
{
    int magicNumber;
    int numberOfItems;
    ImageSize imageSize;
    PixelsOrLabels content;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#pragma endregion

#pragma region File Processing

int castToInt(const STLArray4b &bytes, bool isBigEndian = false)
{
    int value = 0;
    for (STLArray4b::size_type i = 0; i < bytes.size(); i++)
    {
        if (isBigEndian)
        {
            value |= (std::to_integer<int>(bytes[i]) << (3 - i) * 8);
        }
        else
        {
            value |= (std::to_integer<int>(bytes[i]) << i * 8);
        }
    }
    return value;
}

FileBody parseMNISTFile(const fs::path &path)
{
    std::ifstream ifs(path.generic_string(), std::ios::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    STLArray4b bytes;
    auto sizeOfFrame = bytes.size();

    ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
    int magicNumber = castToInt(bytes, true);

    ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
    int numberOfItems = castToInt(bytes, true);

    ImageSize imageSize;
    int cols = 1;
    if (magicNumber == IMAGE_DATA_ID)
    {
        ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
        imageSize.height = castToInt(bytes, true);

        ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
        imageSize.width = castToInt(bytes, true);

        // flatten image
        cols = imageSize.size();
    }

    int totalPixelsOrLabels = numberOfItems * cols;
    std::vector<std::byte> buffer(totalPixelsOrLabels);
    ifs.read(std::bit_cast<char *>(buffer.data()), totalPixelsOrLabels);

    ifs.close();

    if (magicNumber == IMAGE_DATA_ID)
    {
        // TODO: std::vector will be freed when returned
        // notice: this will copy new data
        MatrixXb tmp = MatrixXb::Map(buffer.data(), numberOfItems, cols);
        return {magicNumber, numberOfItems, imageSize, tmp};
    }
    else
    {
        // notice: this will copy new data
        VectorXb tmp = VectorXb::Map(buffer.data(), numberOfItems, cols);
        return {magicNumber, numberOfItems, imageSize, tmp};
    }
}

#pragma endregion

#pragma region Custom Functions

std::pair<Eigen::MatrixXi, Eigen::VectorXi> preprocess(const FileBody &imagesFile, const FileBody &labelsFile)
{
    Eigen::MatrixXi images = std::get<MatrixXb>(imagesFile.content).cast<int>();
    Eigen::VectorXi labels = std::get<VectorXb>(labelsFile.content).cast<int>();
    return std::make_pair((images.array() >= 128).cast<int>().matrix(), labels);
}

void runEStep(const Eigen::MatrixXi &images, Eigen::Ref<MatrixX10d> responsibilities, const Eigen::Ref<Vector10d> &lambda, const Eigen::Ref<Matrix10Xd> &theta)
{
    auto n = images.rows();
    auto groups = theta.rows();

#pragma omp parallel for num_threads(NUM_THREADS)
    for (Eigen::Index i = 0; i < n; i++)
    {
        auto image = images.row(i).array().cast<double>();

        Vector10d tmp;
#pragma unroll
        for (Eigen::Index j = 0; j < groups; j++)
        {
            auto row = theta.row(j).array();
            tmp[j] = (row.pow(image) * (1 - row).pow(1 - image)).prod();
        }

        // lambda_j * P(x_n | theta_j)
        tmp.array() *= lambda.array();

        auto dominator = tmp.sum();
        if (dominator == 0)
        {
            continue;
        }

        responsibilities.row(i) = tmp.array() / dominator;
    }
}

void runMStep(const Eigen::MatrixXi &images, const Eigen::Ref<MatrixX10d> &responsibilities, Eigen::Ref<Vector10d> lambda, Eigen::Ref<Matrix10Xd> theta)
{
    Vector10d totalResponsibilitiesPerGroup = responsibilities.colwise().sum().transpose();
    lambda = (totalResponsibilitiesPerGroup.array() / responsibilities.rows()).matrix();
    theta = totalResponsibilitiesPerGroup.array().max(1).inverse().matrix().asDiagonal() * (responsibilities.transpose() * images.cast<double>());
}

void printClasses(const ImageSize &imageSize, const Eigen::Ref<Matrix10Xd> &theta)
{
    int count = 0;
    for (auto group : theta.rowwise())
    {
        std::cout << "class " << count << ":" << std::endl;

        std::cout << (group.array() > 0.5).cast<bool>().reshaped<Eigen::RowMajor>(imageSize.height, imageSize.width) << std::endl
                  << std::endl;

        count++;
    }
}

void printLabels(const ImageSize &imageSize, const Eigen::Ref<Vector10i> &mapper, const Eigen::Ref<Matrix10Xd> &theta)
{
    // mapper: label => class

    int count = 0;
    for (auto classIndex : mapper)
    {
        std::cout << "labeled class " << count << ":" << std::endl;

        std::cout << (theta.row(classIndex).array() > 0.5).cast<bool>().reshaped<Eigen::RowMajor>(imageSize.height, imageSize.width) << std::endl
                  << std::endl;

        count++;
    }
}

Eigen::Index printConfusionMatrix(const Eigen::Ref<Eigen::VectorXi> &predictions, const Eigen::VectorXi &labels)
{
    Eigen::Index totalTPCount = 0;
    for (int i = 0; i < 10; i++)
    {
        const auto positiveGTMask = labels.array() == i;
        const auto positivePredictionMask = predictions.array() == i;

        auto tpCount = (positivePredictionMask && positiveGTMask).count();
        auto fpCount = positivePredictionMask.count() - tpCount;
        auto fnCount = positiveGTMask.count() - tpCount;
        auto tnCount = (!positiveGTMask).count() - fpCount;

        std::cout << "Confusion Matrix " << i << ":" << std::endl;
        std::cout << "\t\t\t\tPredict number " << i << "\tPredict not number " << i << std::endl;
        std::cout << "Is number " << i << "\t\t\t\t" << tpCount << "\t\t\t\t" << fnCount << std::endl;
        std::cout << "Isn't number " << i << "\t\t\t" << fpCount << "\t\t\t\t" << tnCount << std::endl;

        std::cout << std::endl;

        std::cout << "Sensitivity (Successfully predict number 1): " << static_cast<long double>(tpCount) / positiveGTMask.count() << std::endl;
        std::cout << "Specificity (Successfully predict not number 1): " << static_cast<long double>(tnCount) / (!positiveGTMask).count() << std::endl;

        if (i != 9)
        {
            std::cout << std::endl
                      << "-----------------------------------------------------------------------" << std::endl
                      << std::endl;
        }

        totalTPCount += tpCount;
    }

    std::cout << std::endl;
    return totalTPCount;
}

Vector10i assignLabels(Eigen::Ref<Eigen::VectorXi> predictions, const Eigen::VectorXi &labels)
{
    Vector10i mapper;
    // [labels, classes]
    Eigen::Matrix<Eigen::Index, 10, 10> table = Eigen::Matrix<Eigen::Index, 10, 10>::Zero();

#pragma omp parallel for num_threads(NUM_THREADS)
    for (Eigen::Index i = 0; i < labels.rows(); i++)
    {
        table(labels[i], predictions[i]) += 1;
    }

    int labelId;
    int classId;
    Eigen::VectorXi tmp = predictions;
    for (int i = 0; i < 10; i++)
    {
        table.maxCoeff(&labelId, &classId);

        // avoid assigning the same label twice
        table.row(labelId).fill(-1);
        table.col(classId).fill(-1);

        // label => class
        mapper[labelId] = classId;

        tmp = (predictions.array() == classId).select(labelId, tmp);
    }
    predictions.swap(tmp);

    return mapper;
}

void modelMNIST(const Eigen::MatrixXi &images, const Eigen::VectorXi &labels, const ImageSize &imageSize)
{
    // i: data
    // j: groups
    // k: pixels

    Vector10d lambda = Vector10d::Constant(0.1);
    RandomGenerator generator;
    Matrix10Xd theta = generator.randu(10, imageSize.size(), 0.25, 0.75);
    theta = theta.rowwise().sum().cwiseInverse().asDiagonal() * theta;

    int count = 0;
    MatrixX10d responsibilities = MatrixX10d::Zero(images.rows(), 10);
    MatrixX10d delta;
    do
    {
        Matrix10Xd dresponsibilities = responsibilities;
        runEStep(images, responsibilities, lambda, theta);
        delta = (dresponsibilities - responsibilities).cwiseAbs();

        runMStep(images, responsibilities, lambda, theta);

        printClasses(imageSize, theta);

        count++;

        auto oldPrecision = std::cout.precision();
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout << "No. of Iteration: " << count << ", Difference: " << delta.sum() << ", Mean: " << delta.mean() << std::endl
                  << std::endl;
        std::cout.precision(oldPrecision);
    } while (delta.mean() >= STOP_APPROXIMATION_THRESHOLD && count < 500);

    std::cout << "-----------------------------------------------------------------------" << std::endl;
    std::cout << "-----------------------------------------------------------------------" << std::endl
              << std::endl;

    Eigen::VectorXi predictions(images.rows());
#pragma omp parallel for num_threads(NUM_THREADS)
    for (Eigen::Index i = 0; i < responsibilities.rows(); i++)
    {
        responsibilities.row(i).maxCoeff(&predictions[i]);
    }

    Vector10i mapper = assignLabels(predictions, labels);

    printLabels(imageSize, mapper, theta);
    auto totalCorrects = printConfusionMatrix(predictions, labels);

    std::cout << "Total iteration to converge: " << count << std::endl;
    std::cout << "Total error rate: " << static_cast<long double>(images.rows() - totalCorrects) / images.rows() << std::endl;
}

#pragma endregion

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data path>" << std::endl;
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

    // parse MNIST files
    auto trainImages = parseMNISTFile(path / TRAIN_IMAGES_FILE);
    auto trainLabels = parseMNISTFile(path / TRAIN_LABELS_FILE);

    auto [images, labels] = preprocess(trainImages, trainLabels);
    modelMNIST(images, labels, trainImages.imageSize);
    return 0;
}
