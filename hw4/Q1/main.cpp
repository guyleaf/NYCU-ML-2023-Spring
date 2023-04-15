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

#define TITLE "ML HW4"

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-5;

constexpr ImVec2 DATA_RANGES = ImVec2(-4.5, 14.5);

constexpr ImVec4 COLOR_BLACK = ImVec4(0, 0, 0, 1);
constexpr ImVec4 COLOR_RED = ImVec4(1, 0, 0, 1);
constexpr ImVec4 COLOR_BLUE = ImVec4(0, 0, 1, 1);

using dMatrix2d = algebra::Matrix2d<double>;

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

    auto generate()
    {
        auto x = generator::generate_from_normal_distribution(this->parameterForX.mean, this->parameterForX.variance, this->rng);
        auto y = generator::generate_from_normal_distribution(this->parameterForY.mean, this->parameterForY.variance, this->rng);
        return std::make_pair(x, y);
    }

private:
    std::mt19937_64 rng;

    GaussianParameter parameterForX;
    GaussianParameter parameterForY;
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

void showGUI(const dMatrix2d &points, const dMatrix2d &labels, const dMatrix2d &gradientPredictions, const dMatrix2d &newtonPredictions)
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

                    for (std::size_t i = 0; i < points.rows(); i++)
                    {
                        if (labels(i, 0) == 1)
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

                    for (std::size_t i = 0; i < points.rows(); i++)
                    {
                        if (gradientPredictions(i, 0) > 0.5)
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

                    for (std::size_t i = 0; i < points.rows(); i++)
                    {
                        if (newtonPredictions(i, 0) > 0.5)
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

dMatrix2d makeDegisnMatrix(const dMatrix2d& points)
{
    // [1, y, x]
    dMatrix2d designMatrix(points.rows(), 3, 1);

    for (std::size_t i = 0; i < points.rows(); i++)
    {
        designMatrix(i, 2) = points(i, 0);
        designMatrix(i, 1) = points(i, 1);
    }
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

void printWeights(dMatrix2d &weights)
{
    std::cout << "w:" << std::endl;
    for (auto value : weights)
    {
        std::cout << formatValueOutput(value) << std::endl;
    }
    std::cout << std::endl;
}

void printConfusionMatrix(int numberOfDataPoints, const dMatrix2d &labels, const dMatrix2d &predictions)
{
    int tpCount = 0;
    int tnCount = 0;
    int fpCount = 0;
    int fnCount = 0;

    for (std::size_t i = 0; i < labels.rows(); i++)
    {
        if (std::abs(labels(i, 0) - predictions(i, 0)) < 0.5)
        {
            if (labels(i, 0) == 1)
            {
                tpCount++;
            }
            else
            {
                tnCount++;
            }
        }
        else
        {
            if (labels(i, 0) == 1)
            {
                fnCount++;
            }
            else
            {
                fpCount++;
            }
        }
    }

    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "\t\t\tPredict cluster 1\tPredict cluster 2" << std::endl;
    std::cout << "Is cluster 1\t\t" << tpCount << "\t" << fnCount << std::endl;
    std::cout << "Is cluster 1\t\t" << fpCount << "\t" << tnCount << std::endl;

    std::cout << std::endl;

    std::cout << "Sensitivity (Successfully predict cluster 1): " << static_cast<double>(tpCount) / numberOfDataPoints << std::endl;
    std::cout << "Specificity (Successfully predict cluster 2): " << static_cast<double>(tnCount) / numberOfDataPoints << std::endl;
}

auto generateDataPoints(DataGenerator &d1Generator, DataGenerator &d2Generator, int numberOfDataPoints)
{
    int size = numberOfDataPoints * 2;
    dMatrix2d points(size, 2);
    std::vector<double> labels(size, 0);
    std::fill(labels.begin() + numberOfDataPoints, labels.end(), 1);

    for (int i = 0; i < numberOfDataPoints; i++)
    {
        auto [x1, y1] = d1Generator.generate();
        points.row(i) = {x1, y1};
        auto [x2, y2] = d2Generator.generate();
        points.row(i + numberOfDataPoints) = {x2, y2};
    }
    return std::make_pair(points, dMatrix2d(size, labels));
}

void modelDataGenerator(double learningRate, int numberOfDataPoints, DataGenerator &d1Generator, DataGenerator &d2Generator)
{
    std::random_device rd;
    auto rng = std::mt19937_64(rd());

    // TODO: support concatenation and manipulate row/column with another matrix
    auto [dataPoints, labels] = generateDataPoints(d1Generator, d2Generator, numberOfDataPoints);
    auto designMatrix = makeDegisnMatrix(dataPoints);
    auto gradientWeights = algebra::randn<double>(3, 1, rng);
    auto newtonWeights = gradientWeights;

    int count = 0;
    dMatrix2d dWeights;
    do
    {
        dWeights = gradientWeights;

        // sigmoid + linear regression = logistic regression
        auto y = 1 / (1 + (-1 * designMatrix.mm(gradientWeights)).exp());

        auto jacobianMatrix = designMatrix.transpose().mm(y - labels);

        // Steepest gradient descent
        gradientWeights -= learningRate * jacobianMatrix;

        dWeights -= gradientWeights;
        count++;
    } while (dWeights.abs().mean() > STOP_APPROXIMATION_THRESHOLD && count < 10000);

    count = 0;
    do
    {
        dWeights = newtonWeights;

        // sigmoid + linear regression = logistic regression
        auto y = 1 / (1 + (-1 * designMatrix.mm(newtonWeights)).exp());

        auto jacobianMatrix = designMatrix.transpose().mm(y - labels);

        y *= (1 - y);
        y = algebra::diagonal<double>(y.array());
        auto hessianMatrix = designMatrix.transpose().mm(y.mm(designMatrix));

        // Newton's method
        try
        {
            newtonWeights -= hessianMatrix.inverse().mm(jacobianMatrix);
        }
        catch(const std::runtime_error& e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            newtonWeights -= learningRate * jacobianMatrix;
        }

        dWeights -= newtonWeights;
        count++;
    } while (dWeights.abs().mean() > STOP_APPROXIMATION_THRESHOLD && count < 10000);

    auto gradientPredictions = 1 / (1 + (-1 * designMatrix.mm(gradientWeights)).exp());
    auto newtonPredictions = 1 / (1 + (-1 * designMatrix.mm(newtonWeights)).exp());

    std::cout << "Gradient descent" << std::endl << std::endl;

    printWeights(gradientWeights);

    std::cout << std::endl;

    printConfusionMatrix(numberOfDataPoints, labels, gradientPredictions);

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << "Newton's method" << std::endl << std::endl;

    printWeights(newtonWeights);

    std::cout << std::endl;

    printConfusionMatrix(numberOfDataPoints, labels, newtonPredictions);

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
