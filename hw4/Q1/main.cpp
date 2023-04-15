#include <random>
#include <string>
#include <iostream>
#include <limits>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

#include <algebra/algebra.h>
#include <algebra/matrix.hpp>
#include <generator/generator.h>

#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <Eigen/Core>
#include <Eigen/LU>

#define TITLE "ML HW4"

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-5;

constexpr ImVec2 DATA_RANGES = ImVec2(-4.5, 14.5);

constexpr ImVec4 COLOR_BLACK = ImVec4(0, 0, 0, 1);
constexpr ImVec4 COLOR_RED = ImVec4(1, 0, 0, 1);
constexpr ImVec4 COLOR_BLUE = ImVec4(0, 0, 1, 1);

#pragma region Data Structures

struct GaussianParameter
{
    double mean;
    double variance;
};

class DataGenerator
{
public:
    DataGenerator(const GaussianParameter &parameterForX, const GaussianParameter &parameterForY) : parameterForX(parameterForX), parameterForY(parameterForY)
    {
        std::random_device rd;
        this->rng = std::mt19937_64(rd());
    }

    Eigen::Vector2d generate()
    {
        auto x = this->generate_from_normal_distribution(this->parameterForX.mean, this->parameterForX.variance);
        auto y = this->generate_from_normal_distribution(this->parameterForY.mean, this->parameterForY.variance);
        return Eigen::Vector2d(x, y);
    }

private:
    std::mt19937_64 rng;

    GaussianParameter parameterForX;
    GaussianParameter parameterForY;

    double generate_from_normal_distribution(double m, double var)
    {
        Eigen::MatrixXd samples = this->randu(12, 1, 0.0, 1.0);
        return m + std::sqrt(var) * (samples.sum() - 6);
    }

    Eigen::MatrixXd randu(std::size_t n, std::size_t m, double a, double b)
    {
        std::uniform_real_distribution uniform_dist(a, b);
        auto uniform = [&uniform_dist, &rng = this->rng] () {return uniform_dist(rng);};
        return Eigen::MatrixXd::NullaryExpr(n, m, uniform);
    }
};

#pragma endregion

#pragma region GUI Functions

static void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

GLFWwindow *setUpGUI()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        return nullptr;
    }

    // Decide GL+GLSL versions
    // GL 3.3 + GLSL 330
    const char *glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // 3.0+ only

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(1280, 720, TITLE, nullptr, nullptr);
    if (window == nullptr)
        return nullptr;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();

    ImFontConfig cfg;
    cfg.SizePixels = 15;
    io.Fonts->AddFontDefault(&cfg);

    // Setup Dear ImGui style
    ImGui::StyleColorsLight();
    // ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    return window;
}

void showGUI(const Eigen::MatrixX2d &points, const Eigen::VectorXd &labels, const Eigen::VectorXd &gradientPredictions, const Eigen::VectorXd &newtonPredictions)
{
    auto window = setUpGUI();
    if (window == nullptr)
    {
        throw std::runtime_error("Cannot create window.");
    }

    // Our state
    auto clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            const auto windowSize = ImGui::GetIO().DisplaySize;

            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(windowSize.x, windowSize.y), ImGuiCond_Always);
            ImGui::Begin("Result", nullptr, ImGuiWindowFlags_NoDecoration);

            if (ImPlot::BeginSubplots("Result", 1, 3, ImVec2(windowSize.x, windowSize.y)))
            {
                if (ImPlot::BeginPlot("Ground truth"))
                {
                    ImPlot::SetupAxes("x", "y");
                    ImPlot::SetupAxisLimits(ImAxis_X1, DATA_RANGES.x, DATA_RANGES.y);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -3, 14);

                    for (Eigen::Index i = 0; i < points.rows(); i++)
                    {
                        if (labels[i] == 1)
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_BLUE, IMPLOT_AUTO, COLOR_BLUE);
                            ImPlot::PlotScatter("1", &points(i, 0), &points(i, 1), 1);
                        }
                        else
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_RED, IMPLOT_AUTO, COLOR_RED);
                            ImPlot::PlotScatter("0", &points(i, 0), &points(i, 1), 1);
                        }
                    }
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Gradient descent"))
                {
                    ImPlot::SetupAxes("x", "y");
                    ImPlot::SetupAxisLimits(ImAxis_X1, DATA_RANGES.x, DATA_RANGES.y);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -3, 14);

                    for (Eigen::Index i = 0; i < points.rows(); i++)
                    {
                        if (gradientPredictions[i] > 0.5)
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_BLUE, IMPLOT_AUTO, COLOR_BLUE);
                            ImPlot::PlotScatter("1", &points(i, 0), &points(i, 1), 1);
                        }
                        else
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_RED, IMPLOT_AUTO, COLOR_RED);
                            ImPlot::PlotScatter("0", &points(i, 0), &points(i, 1), 1);
                        }
                    }
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Newton's method"))
                {
                    ImPlot::SetupAxes("x", "y");
                    ImPlot::SetupAxisLimits(ImAxis_X1, DATA_RANGES.x, DATA_RANGES.y);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -3, 14);

                    for (Eigen::Index i = 0; i < points.rows(); i++)
                    {
                        if (newtonPredictions[i] > 0.5)
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_BLUE, IMPLOT_AUTO, COLOR_BLUE);
                            ImPlot::PlotScatter("1", &points(i, 0), &points(i, 1), 1);
                        }
                        else
                        {
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_RED, IMPLOT_AUTO, COLOR_RED);
                            ImPlot::PlotScatter("0", &points(i, 0), &points(i, 1), 1);
                        }
                    }
                    ImPlot::EndPlot();
                }
                ImPlot::EndSubplots();
            }

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

#pragma endregion

#pragma region Custom Functions

Eigen::MatrixX3d makeDegisnMatrix(const Eigen::MatrixX2d& points)
{
    // [1, x, y]
    Eigen::MatrixX3d designMatrix(points.rows(), 3);
    designMatrix << Eigen::VectorXd::Ones(points.rows()), points;
    return designMatrix;
}

std::string formatValueOutput(double value)
{
    std::stringstream ss;
    ss << "\t";
    if (std::signbit(value))
    {
        ss << "-";
    }
    else
    {
        ss << " ";
    }

    ss << std::abs(value);
    return ss.str();
}

void printWeights(const Eigen::Vector3d &weights)
{
    std::cout << "w:" << std::endl;
    std::cout << weights << std::endl;
    // for (auto value : weights)
    // {
    //     std::cout << formatValueOutput(value) << std::endl;
    // }
    std::cout << std::endl;
}

void printConfusionMatrix(const Eigen::VectorXd &labels, const Eigen::VectorXd &predictions)
{
    const auto positiveGTMask = labels.array() == 1;
    const auto positivePredMask = predictions.array() > 0.5;

    auto tpCount = (positiveGTMask && positivePredMask).count();
    auto fpCount = positivePredMask.count() - tpCount;
    auto fnCount = positiveGTMask.count() - tpCount;
    auto tnCount = (!positiveGTMask).count() - fpCount;

    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "\t\t\tPredict cluster 1\tPredict cluster 2" << std::endl;
    std::cout << "Is cluster 1\t\t" << tpCount << "\t\t\t\t" << fnCount << std::endl;
    std::cout << "Is cluster 1\t\t" << fpCount << "\t\t\t\t" << tnCount << std::endl;

    std::cout << std::endl;

    std::cout << "Sensitivity (Successfully predict cluster 1): " << static_cast<long double>(tpCount) / positiveGTMask.count() << std::endl;
    std::cout << "Specificity (Successfully predict cluster 2): " << static_cast<long double>(tnCount) / (!positiveGTMask).count() << std::endl;
}

auto generateDataPoints(DataGenerator &d1Generator, DataGenerator &d2Generator, int numberOfDataPoints)
{
    Eigen::MatrixX2d d1Points(numberOfDataPoints, 2);
    for (auto row: d1Points.rowwise())
    {
        row = d1Generator.generate();
    }

    Eigen::MatrixX2d d2Points(numberOfDataPoints, 2);
    for (auto row: d2Points.rowwise())
    {
        row = d2Generator.generate();
    }

    Eigen::MatrixX2d points(numberOfDataPoints * 2, 2);
    points << d1Points, d2Points;

    Eigen::VectorXd labels(numberOfDataPoints * 2);
    labels << Eigen::VectorXd::Zero(numberOfDataPoints), Eigen::VectorXd::Ones(numberOfDataPoints);

    return std::make_pair(points, labels);
}

void modelDataGenerator(double learningRate, int numberOfDataPoints, DataGenerator &d1Generator, DataGenerator &d2Generator)
{
    // TODO: support concatenation and manipulate row/column with another matrix
    auto [dataPoints, labels] = generateDataPoints(d1Generator, d2Generator, numberOfDataPoints);
    Eigen::MatrixX3d designMatrix = makeDegisnMatrix(dataPoints);
    Eigen::Vector3d gradientWeights = Eigen::Vector3d::Random();
    Eigen::Vector3d newtonWeights = gradientWeights;

    int count = 0;
    Eigen::Vector3d dWeights;
    do
    {
        dWeights = gradientWeights;

        // sigmoid + linear regression = logistic regression
        Eigen::VectorXd y = ((-1 * designMatrix * gradientWeights).array().exp() + 1).inverse().matrix();

        Eigen::Vector3d jacobianMatrix = designMatrix.transpose() * (y - labels);

        // Steepest gradient descent
        gradientWeights -= learningRate * jacobianMatrix;

        dWeights -= gradientWeights;
        count++;
    } while (dWeights.array().abs().mean() > STOP_APPROXIMATION_THRESHOLD && count < 10000);

    count = 0;
    do
    {
        dWeights = newtonWeights;

        // sigmoid + linear regression = logistic regression
        Eigen::VectorXd y = ((-1 * designMatrix * newtonWeights).array().exp() + 1).inverse().matrix();

        Eigen::Vector3d jacobianMatrix = designMatrix.transpose() * (y - labels);

        y = y.array() * (1 - y.array());

        Eigen::DiagonalMatrix<double, Eigen::Dynamic> D(y);
        Eigen::Matrix3d hessianMatrix = designMatrix.transpose() * D * designMatrix;

        // Newton's method
        if (const Eigen::FullPivLU<Eigen::Matrix3d> &fullPivLu = hessianMatrix.fullPivLu(); fullPivLu.isInvertible())
        {
            newtonWeights -= hessianMatrix.inverse() * jacobianMatrix;
        }
        else
        {
            newtonWeights -= learningRate * jacobianMatrix;
        }

        dWeights -= newtonWeights;
        count++;
    } while (dWeights.array().abs().mean() > STOP_APPROXIMATION_THRESHOLD && count < 10000);

    Eigen::VectorXd gradientPredictions = ((-1 * designMatrix * gradientWeights).array().exp() + 1).inverse().matrix();
    Eigen::VectorXd newtonPredictions = ((-1 * designMatrix * newtonWeights).array().exp() + 1).inverse().matrix();

    std::cout << "Gradient descent" << std::endl << std::endl;

    printWeights(gradientWeights);

    std::cout << std::endl;

    printConfusionMatrix(labels, gradientPredictions);

    std::cout << std::endl << "------------------------------------------------------------" << std::endl;
    std::cout << "Newton's method" << std::endl << std::endl;

    printWeights(newtonWeights);

    std::cout << std::endl;

    printConfusionMatrix(labels, newtonPredictions);

    showGUI(dataPoints, labels, gradientPredictions, newtonPredictions);
}

#pragma endregion

int main(int argc, char *argv[])
{
    if (argc < 11)
    {
        std::cerr << "Usage: " << argv[0] << " <learningRate> <number of data points> <mean x1> <variance x1> <mean y1> <variance y1>..." << std::endl;
        return 1;
    }

    argv++;

    auto learningRate = std::stod(*argv);
    argv++;

    auto numberOfDataPoints = std::stoi(*argv);
    argv++;

    std::array<GaussianParameter, 4> gaussianParameters;
    for (auto &parameter : gaussianParameters)
    {
        parameter.mean = std::stod(*argv);
        argv++;
        parameter.variance = std::stod(*argv);
        argv++;
    }

    DataGenerator d1Generator(gaussianParameters[0], gaussianParameters[1]);
    DataGenerator d2Generator(gaussianParameters[2], gaussianParameters[3]);
    modelDataGenerator(learningRate, numberOfDataPoints, d1Generator, d2Generator);

    return 0;
}
