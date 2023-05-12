#include <iostream>
#include <iterator>
#include <vector>
#include <ranges>

#include <boost/filesystem.hpp>
#include <svm.h>
#include <omp.h>

#define NUM_THREADS 12

namespace fs = boost::filesystem;

const std::string TRAIN_X_FILE = "X_train.csv";
const std::string TRAIN_Y_FILE = "Y_train.csv";
const std::string TEST_X_FILE = "X_test.csv";
const std::string TEST_Y_FILE = "Y_test.csv";

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
#pragma omp simd
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

        auto tmp = new svm_node[features.size()];
        std::ranges::copy(features, tmp);
        problem.x[i] = tmp;
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

    return svm_train(&problem, &parameter);
}

std::vector<double> predict(const svm_model &model, const svm_problem &problem)
{
    std::vector<double> predictions(problem.l);
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

void solve(const svm_problem &trainProblem, const svm_problem &testProblem, int numberOfFeatures)
{
    auto linearParameter = createSVMParameter(numberOfFeatures);
    auto linearModel = train(trainProblem, linearParameter);

    auto polyParameter = createSVMParameter(numberOfFeatures);
    polyParameter.kernel_type = POLY;
    auto polyModel = train(trainProblem, polyParameter);

    auto rbfParameter = createSVMParameter(numberOfFeatures);
    rbfParameter.kernel_type = RBF;
    auto rbfModel = train(trainProblem, rbfParameter);

    // Part 1

    std::cout << "Linear Model" << std::endl;

    auto predictions = predict(*linearModel, testProblem);
    evaluate(testProblem, predictions);

    std::cout << "Polynomial Model" << std::endl;

    predictions = predict(*polyModel, testProblem);
    evaluate(testProblem, predictions);

    std::cout << "RBF Model" << std::endl;

    predictions = predict(*rbfModel, testProblem);
    evaluate(testProblem, predictions);

    // Part 2


    // Part 3

    svm_free_and_destroy_model(&linearModel);
    svm_free_and_destroy_model(&polyModel);
    svm_free_and_destroy_model(&rbfModel);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data path>" << std::endl;
        return 1;
    }

    argv++;

    fs::path path(*argv);
    if (!fs::is_directory(path))
    {
        std::cerr << "The data path is not a directory or not existed." << std::endl;
        return 1;
    }

    omp_set_num_threads(NUM_THREADS);

    auto trainXData = parseXFile(path / TRAIN_X_FILE);
    auto trainYData = parseYFile(path / TRAIN_Y_FILE);
    auto testXData = parseXFile(path / TEST_X_FILE);
    auto testYData = parseYFile(path / TEST_Y_FILE);

    auto trainProblem = makeProblem(trainXData, trainYData);
    auto testProblem = makeProblem(testXData, testYData);

    auto numberOfFeatures = static_cast<int>(trainXData[0].size());
    solve(trainProblem, testProblem, numberOfFeatures);

    releaseProblem(trainProblem);
    releaseProblem(testProblem);
    return 0;
}