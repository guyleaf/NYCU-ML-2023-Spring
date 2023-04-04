#include <random>
#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <limits>
#include <sstream>
#include <iomanip>

#include <algebra/algebra.h>
#include <algebra/matrix.hpp>
#include <generator/generator.h>

#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#define TITLE "ML HW3"

constexpr double STOP_APPROXIMATION_THRESHOLD = 1e-3;

constexpr ImVec2 DATA_RANGES = ImVec2(-2, 2);
constexpr int DATA_COUNTS = 200;

constexpr ImVec4 COLOR_BLACK = ImVec4(0, 0, 0, -1);
constexpr ImVec4 COLOR_RED = ImVec4(255, 0, 0, -1);

using dMatrix2d = algebra::Matrix2d<double>;

dMatrix2d makeDegisnMatrix(int basis, double x);
dMatrix2d getPointsFromFunction(const dMatrix2d &weights, double bias = 0, double start = -2, double end = 2, int count = 100);

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
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    return window;
}

void showGUI(const dMatrix2d &samples, const std::array<dMatrix2d, 3> &groundTruthPoints, const std::array<std::array<dMatrix2d, 3>, 3> &predictions)
{
    auto window = setUpGUI();
    if (window == nullptr)
    {
        throw std::runtime_error("Cannot create window.");
    }

    // Our state
    auto clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    auto point_color = ImVec4(255, 0, 0, 1);

    const std::array<int, 3> predictionConditions{static_cast<int>(samples.rows()), 10, 50};

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

            if (ImPlot::BeginSubplots("Result", 2, 2, ImVec2(windowSize.x, windowSize.y)))
            {
                if (ImPlot::BeginPlot("Ground truth"))
                {
                    ImPlot::SetupAxes("x", "y");
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, DATA_RANGES.x, DATA_RANGES.y);
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -25, 25);

                    ImPlot::SetNextLineStyle(COLOR_RED);
                    ImPlot::PlotLine("f_var1(x)", &groundTruthPoints[0].col(0)[0], &groundTruthPoints[0].col(1)[0], groundTruthPoints[0].rows());

                    ImPlot::SetNextLineStyle(COLOR_BLACK);
                    ImPlot::PlotLine("f_mean(x)", &groundTruthPoints[1].col(0)[0], &groundTruthPoints[1].col(1)[0], groundTruthPoints[1].rows());

                    ImPlot::SetNextLineStyle(COLOR_RED);
                    ImPlot::PlotLine("f_var2(x)", &groundTruthPoints[2].col(0)[0], &groundTruthPoints[2].col(1)[0], groundTruthPoints[2].rows());
                    ImPlot::EndPlot();
                }

                for (std::size_t i = 0; i < predictionConditions.size(); i++)
                {
                    int counts = predictionConditions[i];
                    const auto &prediction = predictions[i];

                    std::string title = "After " + std::to_string(counts) + " incomes";
                    if (i == 0)
                    {
                        title = "Predict result";
                    }

                    if (ImPlot::BeginPlot(title.c_str()))
                    {
                        ImPlot::SetupAxes("x", "y");
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, DATA_RANGES.x, DATA_RANGES.y);
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -25, 25);

                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, point_color, IMPLOT_AUTO, point_color);
                        ImPlot::PlotScatter("data", &samples.col(0)[0], &samples.col(1)[0], counts);

                        ImPlot::SetNextLineStyle(COLOR_RED);
                        ImPlot::PlotLine("f_var1(x)", &prediction[0].col(0)[0], &prediction[0].col(1)[0], prediction[0].rows());

                        ImPlot::SetNextLineStyle(COLOR_BLACK);
                        ImPlot::PlotLine("f_mean(x)", &prediction[1].col(0)[0], &prediction[1].col(1)[0], prediction[1].rows());

                        ImPlot::SetNextLineStyle(COLOR_RED);
                        ImPlot::PlotLine("f_var2(x)", &prediction[2].col(0)[0], &prediction[2].col(1)[0], prediction[2].rows());
                        ImPlot::EndPlot();
                    }
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

dMatrix2d getPointsFromFunction(const dMatrix2d &weights, double bias, double start, double end, int count)
{
    dMatrix2d points(count, 2);
    auto n = weights.rows();
    double scale = (end - start) / static_cast<double>(count);
    for (int i = 0; i < count; i++)
    {
        points(i, 0) = start;

        auto designMatrix = makeDegisnMatrix(n, start);
        points(i, 1) = (designMatrix * weights).item() + bias;

        start += scale;
    }

    return points;
}

class DataGenerator
{
public:
    DataGenerator(const dMatrix2d &weights, double variance) : weights(weights), variance(variance)
    {
        std::random_device rd;
        rng = std::mt19937_64(rd());
    }

    auto generate()
    {
        return generator::generate_point_from_ploynomial_basis(this->weights, this->variance, this->rng);
    }

    const dMatrix2d weights;
    const double variance;

private:
    std::mt19937_64 rng;
};

dMatrix2d makeDegisnMatrix(int basis, double x)
{
    dMatrix2d designMatrix(1, basis, 1);
    for (int i = 1; i < basis; i++)
    {
        designMatrix(0, i) = std::pow(x, i);
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

void printPosteriorParameters(dMatrix2d &mean, dMatrix2d &variance)
{
    std::cout << "Postirior mean:" << std::endl;
    for (auto value : mean)
    {
        std::cout << formatValueOutput(value) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Posterior variance:" << std::endl;
    auto rows = variance.rows();
    auto cols = variance.cols();
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < cols; j++)
        {
            std::cout << std::setw(16) << std::left << formatValueOutput(variance(i, j));
            if (j != cols - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::pair<double, double> calculatePredictiveParameters(const dMatrix2d &designMatrix, const dMatrix2d &priorMean, const dMatrix2d &priorVariance, double likelihoodVariance)
{
    double mean = (designMatrix * priorMean).item();
    double variance = likelihoodVariance + (designMatrix * priorVariance * designMatrix.transpose()).item();
    return std::make_pair(mean, variance);
}

void printPredictiveParameters(double mean, double variance)
{
    std::cout << "Predictive distribution ~ N(" << mean << ", " << variance << ")" << std::endl;
    std::cout << std::endl;
}

std::array<dMatrix2d, 3> calculatePredictiveLines(const dMatrix2d &priorMean, const dMatrix2d &priorVariance, double likelihoodVariance, int basis)
{
    auto points = getPointsFromFunction(priorMean, 0, DATA_RANGES.x, DATA_RANGES.y, DATA_COUNTS);

    std::array<dMatrix2d, 3> lines;
    lines.fill(points);

    for (std::size_t i = 0; i < points.rows(); i++)
    {
        auto x = points(i, 0);
        auto y = points(i, 1);

        auto designMatrix = makeDegisnMatrix(basis, x);
        double variance = likelihoodVariance + (designMatrix * priorVariance * designMatrix.transpose()).item();
        lines[0](i, 1) = y + variance;
        lines[2](i, 1) = y - variance;
    }

    return lines;
}

void modelDataGenerator(DataGenerator &generator, int basis, double precisionForInitialPrior)
{
    auto priorMean = algebra::zeros<double>(basis, 1);
    auto priorVariance = algebra::eye<double>(basis, 1 / precisionForInitialPrior);
    auto a = (1 / generator.variance);
    auto b = precisionForInitialPrior;

    std::vector<double> samples;

    std::array<dMatrix2d, 3> groundTruthPoints{
        getPointsFromFunction(generator.weights, generator.variance, DATA_RANGES.x, DATA_RANGES.y, DATA_COUNTS),
        getPointsFromFunction(generator.weights, 0, DATA_RANGES.x, DATA_RANGES.y, DATA_COUNTS),
        getPointsFromFunction(generator.weights, -generator.variance, DATA_RANGES.x, DATA_RANGES.y, DATA_COUNTS)};

    std::array<std::array<dMatrix2d, 3>, 3> predictions;

    std::size_t numberOfSamples = 0;
    dMatrix2d dMean;
    dMatrix2d dVariance;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    do
    {
        dMean = priorMean;
        dVariance = priorVariance;

        auto [x, y] = generator.generate();
        samples.push_back(x);
        samples.push_back(y);
        numberOfSamples++;

        std::cout << "Add data point (" << x << ", " << y << "):" << std::endl
                  << std::endl;

        auto designMatrix = makeDegisnMatrix(basis, x);

        auto gramMatrix = designMatrix.transpose() * designMatrix;

        priorVariance = a * gramMatrix + algebra::eye<double>(basis, b);
        priorVariance = priorVariance.inverse();
        priorMean = a * priorVariance * designMatrix.transpose() * y;

        printPosteriorParameters(priorMean, priorVariance);
        auto [mean, variance] = calculatePredictiveParameters(designMatrix, priorMean, priorVariance, generator.variance);
        printPredictiveParameters(mean, variance);

        if (numberOfSamples == 10)
        {
            predictions[1] = calculatePredictiveLines(priorMean, priorVariance, generator.variance, basis);
        }
        else if (numberOfSamples == 50)
        {
            predictions[2] = calculatePredictiveLines(priorMean, priorVariance, generator.variance, basis);
        }

        dMean = dMean - priorMean;
        dVariance = dVariance - priorVariance;
    } while (dMean.abs().mean() > STOP_APPROXIMATION_THRESHOLD || dVariance.abs().mean() > STOP_APPROXIMATION_THRESHOLD);

    predictions[0] = calculatePredictiveLines(priorMean, priorVariance, generator.variance, basis);

    showGUI(dMatrix2d{numberOfSamples, samples}, groundTruthPoints, predictions);
}

int main(int argc, char *argv[])
{
    if (argc <= 5)
    {
        std::cerr << "Usage: " << argv[0] << " <precision for prior> <basis> <precision for model> <weights>" << std::endl;
        return 1;
    }

    double precisionForInitialPrior = std::stod(argv[1]);

    int basis = std::stoi(argv[2]);
    double variance = std::stod(argv[3]);
    dMatrix2d weights(basis, 1);
    for (std::size_t i = 0; i < basis; i++)
    {
        weights(i, 0) = std::stod(argv[4 + i]);
    }

    DataGenerator generator(weights, variance);
    modelDataGenerator(generator, basis, precisionForInitialPrior);

    return 0;
}