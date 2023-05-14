#include <iostream>
#include <iterator>
#include <vector>
#include <valarray>
#include <algorithm>
#include <cmath>

#include <boost/filesystem.hpp>
#include <svm.h>
#include <omp.h>

#define NUM_THREADS 12

namespace fs = boost::filesystem;

const std::string TRAIN_X_FILE = "X_train.csv";
const std::string TRAIN_Y_FILE = "Y_train.csv";
const std::string TEST_X_FILE = "X_test.csv";
const std::string TEST_Y_FILE = "Y_test.csv";

#pragma region Data Structures

template <typename T>
struct GridSearchRange
{
    T start = -1;
    T end = -1;
    T step = 1;
};

struct GridSearchSettings
{
    /* for poly */
    GridSearchRange<int> degree;
    GridSearchRange<double> coef0;

    /* for poly/rbf */
    GridSearchRange<double> gamma;

    /* for C_SVC */
    GridSearchRange<double> C;

    int kFold = 3;
};

#pragma endregion

#pragma region File Processing

std::vector<std::vector<double>> parseXFile(const fs::path &path)
{
    std::ifstream ifs(path.generic_string());
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    double tmp;
    std::vector<double> row;
    std::vector<std::vector<double>> data;
    while (!ifs.eof())
    {
        ifs >> tmp;
        row.push_back(tmp);

        if (ifs.get() == '\n')
        {
            data.push_back(row);
            row.clear();
        }
    }

    ifs.close();
    return data;
}

std::vector<double> parseYFile(const fs::path &path)
{
    std::ifstream ifs(path.generic_string());
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    std::vector<double> data{std::istream_iterator<double>(ifs), std::istream_iterator<double>()};
    ifs.close();
    return data;
}

#pragma endregion

#pragma region Data Preprocessing

svm_problem makeProblem(const std::vector<std::vector<double>> &x, std::vector<double> &y)
{
    auto featureSize = static_cast<int>(x[0].size());
    svm_problem problem;

    problem.l = static_cast<int>(x.size());
    problem.y = y.data();

    problem.x = new svm_node *[problem.l];
    for (int i = 0; i < problem.l; i++)
    {
        problem.x[i] = new svm_node[featureSize];
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < problem.l; i++)
    {
        std::vector<svm_node> features;

        for (int j = 0; j < featureSize; j++)
        {
            double feature = x[i][j];
            if (feature == 0)
            {
                continue;
            }

            features.push_back(svm_node{j + 1, feature});
        }

        // insert end of features
        features.push_back(svm_node{-1, 0});

        std::ranges::move(features, problem.x[i]);
    }

    return problem;
}

double calculateKernel(const std::valarray<double> &x1, const std::valarray<double> &x2, const svm_parameter &parameter)
{
    auto linear = (x1 * x2).sum();
    auto rbf = std::exp(-parameter.gamma * std::pow((x1 - x2), 2).sum());
    return linear + rbf;
}

svm_problem makeKernel(const std::vector<std::vector<double>> &x1, const std::vector<std::vector<double>> &x2, std::vector<double> &y, const svm_parameter &parameter)
{
    svm_problem problem;
    auto numberOfX1Data = static_cast<int>(x1.size());
    auto numberOfX2Data = static_cast<int>(x2.size());

    problem.l = numberOfX1Data;
    problem.y = y.data();

    problem.x = new svm_node *[numberOfX1Data];
    for (int i = 0; i < numberOfX1Data; i++)
    {
        problem.x[i] = new svm_node[numberOfX2Data + 2];
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < numberOfX1Data; i++)
    {
        std::vector<svm_node> features(numberOfX2Data + 2);

        auto xi = std::valarray<double>(x1[i].data(), x1[i].size());

        features[0] = svm_node{0, static_cast<double>(i + 1)};

#pragma omp simd
        for (int j = 0; j < numberOfX2Data; j++)
        {
            features[j + 1] = svm_node{j + 1, calculateKernel(xi, std::valarray<double>(x2[j].data(), x2[j].size()), parameter)};
        }

        features[numberOfX2Data + 1] = svm_node{-1, 0};

        std::ranges::move(features, problem.x[i]);
    }

    return problem;
}

#pragma endregion

void releaseProblem(svm_problem &problem)
{
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < problem.l; i++)
    {
        delete[] problem.x[i];
    }

    delete[] problem.x;
}

svm_parameter createSVMParameter(int numOfFeatures)
{
    svm_parameter parameter;
    parameter.svm_type = C_SVC;
    parameter.kernel_type = LINEAR;
    parameter.degree = 3;
    parameter.gamma = 1 / static_cast<double>(numOfFeatures); // 1/num_features
    parameter.coef0 = 0;
    parameter.nu = 0.5;
    parameter.cache_size = 100;
    parameter.C = 1;
    parameter.eps = 1e-3;
    parameter.p = 0.1;
    parameter.shrinking = 1;
    parameter.probability = 0;
    parameter.nr_weight = 0;
    parameter.weight_label = nullptr;
    parameter.weight = nullptr;
    return parameter;
}

svm_model *train(const svm_problem &problem, const svm_parameter &parameter)
{
    if (auto error = svm_check_parameter(&problem, &parameter); error != nullptr)
    {
        std::cerr << error << std::endl;
        throw std::runtime_error(error);
    }

    std::cout << "Start training..." << std::endl;

    return svm_train(&problem, &parameter);
}

std::vector<double> predict(const svm_model &model, const svm_problem &problem)
{
    std::vector<double> predictions(problem.l);

    std::cout << "Start predicting..." << std::endl;
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < problem.l; i++)
    {
        predictions[i] = svm_predict(&model, problem.x[i]);
    }
    return predictions;
}

double evaluate(const svm_problem &problem, const std::vector<double> &predictions)
{
    int correctCount = 0;

    std::cout << "Start evaluating..." << std::endl;
#pragma omp parallel for reduction(+ : correctCount) num_threads(NUM_THREADS)
    for (int i = 0; i < problem.l; i++)
    {
        if (problem.y[i] == predictions[i])
        {
            correctCount++;
        }
    }

    double accuracy = static_cast<double>(correctCount) / problem.l;
    std::cout << "Accuracy: " << accuracy << std::endl;
    return accuracy;
}

void train_evaluate(const svm_problem &trainProblem, const svm_problem &testProblem, const svm_parameter &parameter)
{
    auto model = train(trainProblem, parameter);
    auto predictions = predict(*model, testProblem);
    evaluate(testProblem, predictions);
    svm_free_and_destroy_model(&model);
}

template <typename T>
std::vector<T> generateSequence(const GridSearchRange<T> &range)
{
    std::vector<T> sequence;

    T i = range.start;
    while (i < range.end)
    {
        sequence.push_back(i);
        i += range.step;
    }
    return sequence;
}

svm_parameter findCSVCParametersByGridSearch(const svm_problem &problem, const svm_parameter &defaultParameter, const GridSearchSettings &settings)
{
    GridSearchSettings _settings = settings;

    if (defaultParameter.kernel_type != PRECOMPUTED && defaultParameter.kernel_type != POLY)
    {
        _settings.degree.start = defaultParameter.degree;
        _settings.degree.end = defaultParameter.degree + 1;
        _settings.degree.step = 1;

        _settings.coef0.start = defaultParameter.coef0;
        _settings.coef0.end = defaultParameter.coef0 + 1;
        _settings.coef0.step = 1;

        if (defaultParameter.kernel_type != RBF)
        {
            _settings.gamma.start = defaultParameter.gamma;
            _settings.gamma.end = defaultParameter.gamma + 1;
            _settings.gamma.step = 1;
        }
    }

    auto degreeList = generateSequence(_settings.degree);
    auto coef0List = generateSequence(_settings.coef0);
    auto gammaList = generateSequence(_settings.gamma);
    auto CList = generateSequence(_settings.C);

    double bestAccuracy = 0;
    svm_parameter bestParameter = defaultParameter;
#pragma omp parallel for collapse(4) num_threads(NUM_THREADS)
    for (auto degree : degreeList)
    {
        for (auto coef0 : coef0List)
        {
            for (auto gamma : gammaList)
            {
                for (auto C : CList)
                {
                    svm_parameter parameter = defaultParameter;
                    parameter.degree = degree;
                    parameter.coef0 = std::pow(2, coef0);
                    parameter.gamma = std::pow(2, gamma);
                    parameter.C = std::pow(2, C);

                    std::vector<double> targets(problem.l);
                    svm_cross_validation(&problem, &parameter, _settings.kFold, targets.data());

                    auto accuracy = evaluate(problem, targets);
#pragma omp critical
                    if (accuracy > bestAccuracy)
                    {
                        bestParameter = parameter;
                        bestAccuracy = accuracy;
                    }
                }
            }
        }
    }

    std::cout << "Best Cross Validation Accuracy: " << bestAccuracy << std::endl;
    std::cout << "Degree: " << bestParameter.degree << std::endl;
    std::cout << "Coef0: " << bestParameter.coef0 << std::endl;
    std::cout << "Gamma: " << bestParameter.gamma << std::endl;
    std::cout << "C: " << bestParameter.C << std::endl;

    return bestParameter;
}

svm_parameter findCSVCParametersByGridSearch(const std::vector<std::vector<double>> &x, std::vector<double> &y, const svm_parameter &defaultParameter, const GridSearchSettings &settings)
{
    auto degreeList = generateSequence(settings.degree);
    auto coef0List = generateSequence(settings.coef0);
    auto gammaList = generateSequence(settings.gamma);
    auto CList = generateSequence(settings.C);

    double bestAccuracy = 0;
    svm_parameter bestParameter = defaultParameter;
#pragma omp parallel for collapse(4) num_threads(NUM_THREADS)
    for (auto degree : degreeList)
    {
        for (auto coef0 : coef0List)
        {
            for (auto gamma : gammaList)
            {
                for (auto C : CList)
                {
                    svm_parameter parameter = defaultParameter;
                    parameter.degree = degree;
                    parameter.coef0 = std::pow(2, coef0);
                    parameter.gamma = std::pow(2, gamma);
                    parameter.C = std::pow(2, C);

                    auto problem = makeKernel(x, x, y, parameter);

                    std::vector<double> targets(problem.l);
                    svm_cross_validation(&problem, &parameter, settings.kFold, targets.data());

                    auto accuracy = evaluate(problem, targets);
                    releaseProblem(problem);

#pragma omp critical
                    if (accuracy > bestAccuracy)
                    {
                        bestParameter = parameter;
                        bestAccuracy = accuracy;
                    }
                }
            }
        }
    }

    std::cout << "Best Cross Validation Accuracy: " << bestAccuracy << std::endl;
    std::cout << "Degree: " << bestParameter.degree << std::endl;
    std::cout << "Coef0: " << bestParameter.coef0 << std::endl;
    std::cout << "Gamma: " << bestParameter.gamma << std::endl;
    std::cout << "C: " << bestParameter.C << std::endl;

    return bestParameter;
}

void solvePart1(const svm_problem &trainProblem, const svm_problem &testProblem, int numberOfFeatures)
{
    auto linearParameter = createSVMParameter(numberOfFeatures);

    auto polyParameter = createSVMParameter(numberOfFeatures);
    polyParameter.kernel_type = POLY;

    auto rbfParameter = createSVMParameter(numberOfFeatures);
    rbfParameter.kernel_type = RBF;

    // Part 1 Defaults
    std::cout << "Part 1" << std::endl;

    std::cout << "Linear Model" << std::endl;
    train_evaluate(trainProblem, testProblem, linearParameter);

    std::cout << "-----------------------------------------------------------------" << std::endl;

    std::cout << "Polynomial Model" << std::endl;
    train_evaluate(trainProblem, testProblem, polyParameter);

    std::cout << "-----------------------------------------------------------------" << std::endl;

    std::cout << "RBF Model" << std::endl;
    train_evaluate(trainProblem, testProblem, rbfParameter);

    std::cout << "==================================================================" << std::endl;
}

void solvePart2(const svm_problem &trainProblem, const svm_problem &testProblem, int numberOfFeatures)
{
    auto linearParameter = createSVMParameter(numberOfFeatures);

    auto polyParameter = createSVMParameter(numberOfFeatures);
    polyParameter.kernel_type = POLY;

    auto rbfParameter = createSVMParameter(numberOfFeatures);
    rbfParameter.kernel_type = RBF;

    // Part 2 Grid Search
    std::cout << "Part 2" << std::endl;

    GridSearchSettings settings;
    settings.degree.start = 1;
    settings.degree.end = 10;
    settings.coef0.start = -10;
    settings.coef0.end = 11;
    settings.gamma.start = -10;
    settings.gamma.end = 11;
    settings.C.start = -10;
    settings.C.end = 11;
    settings.kFold = 5;

    std::cout << "Linear Model" << std::endl;
    linearParameter = findCSVCParametersByGridSearch(trainProblem, linearParameter, settings);
    train_evaluate(trainProblem, testProblem, linearParameter);

    std::cout << "-----------------------------------------------------------------" << std::endl;

    std::cout << "Polynomial Model" << std::endl;
    polyParameter = findCSVCParametersByGridSearch(trainProblem, polyParameter, settings);
    train_evaluate(trainProblem, testProblem, polyParameter);

    std::cout << "-----------------------------------------------------------------" << std::endl;

    std::cout << "RBF Model" << std::endl;
    rbfParameter = findCSVCParametersByGridSearch(trainProblem, polyParameter, settings);
    train_evaluate(trainProblem, testProblem, rbfParameter);

    std::cout << "==================================================================" << std::endl;
}

void solvePart3(const std::vector<std::vector<double>> &trainX, std::vector<double> &trainY, const std::vector<std::vector<double>> &testX, std::vector<double> &testY, int numberOfFeatures)
{
    auto parameter = createSVMParameter(numberOfFeatures);
    parameter.kernel_type = PRECOMPUTED;

    // Part 3 RBF + Linear kernel
    std::cout << "Part 3" << std::endl;

    GridSearchSettings settings;
    settings.degree.start = parameter.degree;
    settings.degree.end = parameter.degree + 1;
    settings.degree.step = 1;

    settings.coef0.start = parameter.coef0;
    settings.coef0.end = parameter.coef0 + 1;
    settings.coef0.step = 1;

    settings.gamma.start = -10;
    settings.gamma.end = 11;
    settings.C.start = -10;
    settings.C.end = 11;
    settings.kFold = 5;

    parameter = findCSVCParametersByGridSearch(trainX, trainY, parameter, settings);

    auto trainProblem = makeKernel(trainX, trainX, trainY, parameter);
    auto testProblem = makeKernel(testX, trainX, testY, parameter);
    train_evaluate(trainProblem, testProblem, parameter);

    std::cout << "==================================================================" << std::endl;
    releaseProblem(trainProblem);
    releaseProblem(testProblem);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <data path> <mode>" << std::endl;
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

    auto mode = std::stoi(*argv);

    omp_set_num_threads(NUM_THREADS);

    auto trainXData = parseXFile(path / TRAIN_X_FILE);
    auto trainYData = parseYFile(path / TRAIN_Y_FILE);
    auto testXData = parseXFile(path / TEST_X_FILE);
    auto testYData = parseYFile(path / TEST_Y_FILE);

    auto numberOfFeatures = static_cast<int>(trainXData[0].size());

    // Don't print anything
    svm_set_print_string_function([](const char *) {});

    switch (mode)
    {
    case 1:
    {
        auto trainProblem = makeProblem(trainXData, trainYData);
        auto testProblem = makeProblem(testXData, testYData);
        std::cout << testProblem.l << std::endl;
        solvePart1(trainProblem, testProblem, numberOfFeatures);
        releaseProblem(trainProblem);
        releaseProblem(testProblem);
        break;
    }
    case 2:
    {
        auto trainProblem = makeProblem(trainXData, trainYData);
        auto testProblem = makeProblem(testXData, testYData);
        solvePart2(trainProblem, testProblem, numberOfFeatures);
        releaseProblem(trainProblem);
        releaseProblem(testProblem);
        break;
    }
    case 3:
        solvePart3(trainXData, trainYData, testXData, testYData, numberOfFeatures);
        break;
    default:
        std::cerr << "Unknown mode." << std::endl;
        return 1;
    }

    return 0;
}