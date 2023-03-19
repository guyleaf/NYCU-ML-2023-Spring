#include <iostream>
#include <string>
#include <fstream>
#include <iterator>
#include <array>
#include <valarray>
#include <algorithm>
#include <numeric>

#define _USE_MATH_DEFINES
#include <cmath>

#include <limits>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include <algebra/matrix.hpp>
#include <common/cast.hpp>

#define TRAIN_IMAGES_FILE "train-images.idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels.idx1-ubyte"
#define TEST_IMAGES_FILE "t10k-images.idx3-ubyte"
#define TEST_LABELS_FILE "t10k-labels.idx1-ubyte"

constexpr int IMAGE_DATA_ID = 2051;
constexpr int LABEL_DATA_ID = 2049;

constexpr unsigned int NUMBER_OF_LABELS = 10;
constexpr unsigned int NUMBER_OF_BINS = 32;

constexpr unsigned int RANGE_OF_GRAY_SCALE = 256;

enum FunctionMode
{
    DISCRETE,
    CONTINUOUS
};

using ByteMatrix = algebra::Matrix2d<unsigned char>;
using DataMatrix = algebra::Matrix2d<double>;

struct ImageSize
{
    int height = 0;
    int width = 0;

    int size() const
    {
        return height * width;
    }
};

struct FileBody
{
    int magicNumber;
    int numberOfItems;
    ImageSize imageSize;
    ByteMatrix content;
};

FileBody parseMNISTFile(fs::path path)
{
    std::ifstream ifs(path.generic_string(), std::ios_base::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    unsigned char bytes[4];
    auto sizeOfFrame = sizeof(bytes);

    ifs.read(reinterpret_cast<char *>(bytes), sizeOfFrame);
    int magicNumber = common::castToInt(bytes, sizeOfFrame, true);

    ifs.read(reinterpret_cast<char *>(bytes), sizeOfFrame);
    int numberOfItems = common::castToInt(bytes, sizeOfFrame, true);

    ImageSize imageSize;
    int cols = 1;
    if (magicNumber == IMAGE_DATA_ID)
    {
        ifs.read(reinterpret_cast<char *>(bytes), sizeOfFrame);
        imageSize.height = common::castToInt(bytes, sizeOfFrame, true);

        ifs.read(reinterpret_cast<char *>(bytes), sizeOfFrame);
        imageSize.width = common::castToInt(bytes, sizeOfFrame, true);

        // flatten image
        cols = imageSize.size();
    }

    std::valarray<unsigned char> buffer(numberOfItems * cols);
    // avoid copy the values to out of buffer
    std::copy_n(std::istreambuf_iterator<char>(ifs), buffer.size(), std::begin(buffer));

    ifs.close();

    return {magicNumber, numberOfItems, imageSize, ByteMatrix(numberOfItems, cols, buffer)};
}

std::array<DataMatrix, NUMBER_OF_LABELS> preprocess(const FileBody &imagesFile, const FileBody &labelsFile, int bins)
{
    const unsigned int sizeOfBin = RANGE_OF_GRAY_SCALE / bins;
    if (sizeOfBin * bins != RANGE_OF_GRAY_SCALE)
    {
        throw std::runtime_error("The range of gray scale is not divisible by bins.");
    }

    const int numberOfPixels = imagesFile.imageSize.size();

    // [labels, pixels, bins]
    std::array<DataMatrix, NUMBER_OF_LABELS> data;
    // [pixels, bins]
    data.fill(DataMatrix(numberOfPixels, bins, 0));

    const auto &images = imagesFile.content;
    const auto &labels = labelsFile.content;
    for (int i = 0; i < labelsFile.numberOfItems; i++)
    {
        auto label = labels(i, 0);
        for (int j = 0; j < numberOfPixels; j++)
        {
            // converted pixel
            auto convertedPixel = images(i, j) / sizeOfBin;
            data[label](j, convertedPixel)++;
        }
    }

    return data;
}

std::array<double, NUMBER_OF_LABELS> calculatePriors(const std::array<DataMatrix, NUMBER_OF_LABELS> &data, int numberOfImages)
{
    const double dominator = numberOfImages + NUMBER_OF_LABELS;

    std::array<double, NUMBER_OF_LABELS> priors;
    for (std::size_t i = 0; i < NUMBER_OF_LABELS; i++)
    {
        // # of images which is label i / total images
        priors[i] = std::log((data[i].row(0).sum() + 1) / dominator);
    }
    return priors;
}

double calculateLikelihoodInDiscreteMode(const std::valarray<unsigned char> &image, const DataMatrix &data)
{
    const auto numberOfPixels = image.size();
    const auto numberOfImages = data.row(0).sum() + 1;

    double result = 0;
    for (std::size_t i = 0; i < numberOfPixels; i++)
    {
        result += std::log((data(i, image[i]) + 1) / numberOfImages);
    }
    return result;
}

void printPosteriors(const std::valarray<double> &posteriors)
{
    const auto originalPrecision = std::cout.precision();
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "Posterior (in log scale):" << std::endl;
    for (std::size_t i = 0; i < posteriors.size(); i++)
    {
        std::cout << i << ": " << posteriors[i] << std::endl;
    }
    std::cout.precision(originalPrecision);
}

void printLabels(const std::array<DataMatrix, NUMBER_OF_LABELS> &data, const ImageSize &imageSize)
{
    std::cout << "Imagination of numbers in Bayesian classifier:" << std::endl
              << std::endl;

    const auto halfOfNumberOfBins = NUMBER_OF_BINS / 2;
    for (std::size_t i = 0; i < NUMBER_OF_LABELS; i++)
    {
        const auto &pixels = data[i];
        const auto numberOfPixels = pixels.rows();
        std::cout << i << ":" << std::endl;
        for (std::size_t j = 0; j < numberOfPixels; j++)
        {
            auto pixel = pixels.row(j);
            auto prediction = std::distance(std::begin(pixel), std::max_element(std::begin(pixel), std::end(pixel)));
            std::cout << (prediction >= halfOfNumberOfBins) << " ";
            if ((j + 1) % imageSize.width == 0)
            {
                std::cout << std::endl;
            }
        }

        std::cout << std::endl;
    }
}

void classifyImagesInDiscreteMode(const FileBody &imagesFile, const std::array<DataMatrix, NUMBER_OF_LABELS> &data, const FileBody &testImagesFile, const FileBody &testLabelsFile)
{
    const std::array<double, NUMBER_OF_LABELS> priors = calculatePriors(data, imagesFile.numberOfItems);

    const auto &images = testImagesFile.content;
    const unsigned int sizeOfBin = RANGE_OF_GRAY_SCALE / NUMBER_OF_BINS;

    double errors = 0;
    std::valarray<double> posteriors(NUMBER_OF_LABELS);
    for (int i = 0; i < testImagesFile.numberOfItems; i++)
    {
        auto image = images.row(i) / sizeOfBin;
        for (std::size_t j = 0; j < NUMBER_OF_LABELS; j++)
        {
            // condition on label j
            posteriors[j] = calculateLikelihoodInDiscreteMode(image, data[j]) + priors[j];
        }

        // use softmax to normalize posteriors
        posteriors = std::exp(posteriors - posteriors.max());
        posteriors /= posteriors.sum();

        printPosteriors(posteriors);

        auto prediction = std::distance(std::begin(posteriors), std::max_element(std::begin(posteriors), std::end(posteriors)));
        auto groundTruth = testLabelsFile.content(i, 0);
        std::cout << "Prediction: "
                  << prediction
                  << ", Ans: "
                  << static_cast<unsigned>(groundTruth) << std::endl
                  << std::endl;

        errors += (prediction != groundTruth);
    }

    printLabels(data, imagesFile.imageSize);

    std::cout << "Error rate: " << errors / testImagesFile.numberOfItems << std::endl;
}

double getLogProbabilityFromGaussianDistribution(double mean, double std, double x)
{
    // log f(x) = -0.5 * ((x - mean) / std)^2 - log (std * sqrt(PI * 2))
    // eps = 1e-2 to avoid division by zero
    std += 1e-2;
    return -0.5 * std::pow((x - mean) / std, 2) - 0.5 * std::log(std::pow(std, 2) * M_PI * 2);
}

std::array<DataMatrix, NUMBER_OF_LABELS> findGaussianParameters(const std::array<DataMatrix, NUMBER_OF_LABELS> &data)
{
    std::array<DataMatrix, NUMBER_OF_LABELS> result;

    const auto numberOfPixels = data[0].rows();
    for (std::size_t i = 0; i < NUMBER_OF_LABELS; i++)
    {
        auto params = DataMatrix(numberOfPixels, 2);

        const auto &pixels = data[i];
        for (std::size_t j = 0; j < numberOfPixels; j++)
        {
            double mean = 0, N = 1;

            for (int scale = 0; scale < RANGE_OF_GRAY_SCALE; scale++)
            {
                auto count = pixels(j, scale);
                mean += (count * (static_cast<double>(scale) / (RANGE_OF_GRAY_SCALE - 1)));
                N += count;
            }
            mean /= N;

            double std = std::pow(-mean, 2);
            for (int scale = 0; scale < RANGE_OF_GRAY_SCALE; scale++)
            {
                auto tmp = pixels(j, scale);
                std += tmp * std::pow((static_cast<double>(scale) / (RANGE_OF_GRAY_SCALE - 1)) - mean, 2);
            }
            std = std::sqrt(std / N);

            params(j, 0) = mean;
            params(j, 1) = std;
        }

        result[i] = params;
    }
    return result;
}

double calculateLikelihoodInContinuousMode(const std::valarray<unsigned char> &image, const DataMatrix &dists)
{
    const auto numberOfPixels = image.size();

    double result = 0;
    for (std::size_t i = 0; i < numberOfPixels; i++)
    {
        auto value = static_cast<double>(image[i]) / (RANGE_OF_GRAY_SCALE - 1);
        result += getLogProbabilityFromGaussianDistribution(dists(i, 0), dists(i, 1), value);
    }
    return result;
}

void printLabelsForGaussian(const std::array<DataMatrix, NUMBER_OF_LABELS> &gaussianParams, const ImageSize &imageSize)
{
    std::cout << "Imagination of numbers in Bayesian classifier:" << std::endl
              << std::endl;

    for (std::size_t i = 0; i < NUMBER_OF_LABELS; i++)
    {
        const auto &params = gaussianParams[i];
        const auto numberOfPixels = params.rows();
        std::cout << i << ":" << std::endl;
        for (std::size_t j = 0; j < numberOfPixels; j++)
        {
            std::cout << (params(j, 0) >= 0.5) << " ";
            if ((j + 1) % imageSize.width == 0)
            {
                std::cout << std::endl;
            }
        }

        std::cout << std::endl;
    }
}

void classifyImagesInContinuousMode(const FileBody &imagesFile, const std::array<DataMatrix, NUMBER_OF_LABELS> &data, const FileBody &testImagesFile, const FileBody &testLabelsFile)
{
    const auto priors = calculatePriors(data, imagesFile.numberOfItems);
    const auto gaussianParams = findGaussianParameters(data);

    const auto &images = testImagesFile.content;

    double errors = 0;
    std::valarray<double> posteriors(NUMBER_OF_LABELS);
    for (int i = 0; i < testImagesFile.numberOfItems; i++)
    {
        auto image = images.row(i);
        for (std::size_t j = 0; j < NUMBER_OF_LABELS; j++)
        {
            // condition on label j
            const auto &params = gaussianParams[j];
            posteriors[j] = calculateLikelihoodInContinuousMode(image, params) + priors[j];
        }

        // use softmax to normalize posteriors
        posteriors = std::exp(posteriors - posteriors.max());
        posteriors /= posteriors.sum();

        printPosteriors(posteriors);

        auto prediction = std::distance(std::begin(posteriors), std::max_element(std::begin(posteriors), std::end(posteriors)));
        auto groundTruth = testLabelsFile.content(i, 0);
        std::cout << "Prediction: "
                  << prediction
                  << ", Ans: "
                  << static_cast<unsigned>(groundTruth) << std::endl
                  << std::endl;

        errors += (prediction != groundTruth);
    }

    printLabelsForGaussian(gaussianParams, imagesFile.imageSize);

    std::cout << "Error rate: " << errors / testImagesFile.numberOfItems << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "The arguments must include {mode} {data path}" << std::endl;
        return 1;
    }

    FunctionMode mode;
    try
    {
        mode = FunctionMode(std::stoi(argv[1]));
    }
    catch (const std::invalid_argument &exception)
    {
        std::cerr << exception.what() << std::endl;
        return 1;
    }

    fs::path path(argv[2]);
    if (!fs::is_directory(path))
    {
        std::cerr << "The data path is not a directory or not existed." << std::endl;
        return 1;
    }

    // parse MNIST files
    auto trainImages = parseMNISTFile(path / TRAIN_IMAGES_FILE);
    auto trainLabels = parseMNISTFile(path / TRAIN_LABELS_FILE);
    auto testImages = parseMNISTFile(path / TEST_IMAGES_FILE);
    auto testLabels = parseMNISTFile(path / TEST_LABELS_FILE);

    // classify images
    switch (mode)
    {
    case FunctionMode::DISCRETE:
    {
        // preprocess images
        // convert pixels to bins & count the values for each pixel.
        auto data = preprocess(trainImages, trainLabels, NUMBER_OF_BINS);
        classifyImagesInDiscreteMode(trainImages, data, testImages, testLabels);
        break;
    }

    case FunctionMode::CONTINUOUS:
    {
        // preprocess images
        // convert pixels to bins & count the values for each pixel.
        auto data = preprocess(trainImages, trainLabels, RANGE_OF_GRAY_SCALE);
        classifyImagesInContinuousMode(trainImages, data, testImages, testLabels);
        break;
    }

    default:
        std::cerr << "Unknown function mode." << std::endl;
        return 1;
    }

    return 0;
}