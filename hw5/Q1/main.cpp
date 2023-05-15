#include <string>
#include <iostream>
#include <vector>
#include <iterator>
#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <fstream>

#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#define NUM_THREADS 12
// #define EIGEN_USE_MKL_ALL

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
const std::string TITLE = "ML HW5 Q1";

constexpr ImVec4 COLOR_BLACK = ImVec4(0, 0, 0, 1);
constexpr ImVec4 COLOR_RED = ImVec4(1, 0, 0, 1);
constexpr ImVec4 COLOR_BLUE = ImVec4(0, 0, 1, 1);

#pragma region Data Structures

using MatrixXd = Eigen::MatrixXd;
using MatrixX2d = Eigen::MatrixX2d;
using MatrixX3d = Eigen::MatrixX3d;

using VectorXd = Eigen::VectorXd;
using Vector3d = Eigen::Vector3d;

struct LossArguments
{
    const MatrixX2d &data;
    double beta;
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
    GLFWwindow *window = glfwCreateWindow(1280, 720, TITLE.c_str(), nullptr, nullptr);
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

void drawPlot(const std::string &title, const MatrixX2d &data, const MatrixX3d &f)
{
    if (ImPlot::BeginPlot(title.c_str()))
    {
        ImPlot::SetupAxes("x", "y");
        ImPlot::SetupAxisLimits(ImAxis_X1, -60, 60);
        // ImPlot::SetupAxisLimits(ImAxis_Y1, -60, 60);

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, COLOR_RED, IMPLOT_AUTO, COLOR_RED);
        ImPlot::PlotScatter("train data", data.col(0).data(), data.col(1).data(), data.rows());

        VectorXd variance = 2 * f.col(2);
        VectorXd upperBound = f.col(1) + variance;
        VectorXd lowerBound = f.col(1) - variance;
        ImPlot::SetNextFillStyle(COLOR_BLUE, 0.5f);
        ImPlot::PlotShaded("f(x)'s 95%% confidence", f.col(0).data(), upperBound.data(), lowerBound.data(), f.rows());

        ImPlot::SetNextLineStyle(COLOR_BLACK);
        ImPlot::PlotLine("f(x)'s mean", f.col(0).data(), f.col(1).data(), f.rows());

        ImPlot::EndPlot();
    }
}

void showGUI(const MatrixX2d &data, const MatrixX3d &f, const MatrixX3d &optimizedF)
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

            if (ImPlot::BeginSubplots("Result", 1, 2, ImVec2(windowSize.x, windowSize.y)))
            {
                drawPlot("Original", data, f);
                drawPlot("Optimized", data, optimizedF);
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
        // in order to use external data in autodiff library
        // create a lambda function to wrap the loss function
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

MatrixX3d generatePointsOfLine(const MatrixX2d &data, double beta, const MatrixXd &covariance, const Vector3d& kernelParameters)
{
    MatrixXd inv_covariance = covariance.inverse();
    double inv_beta = 1 / beta;
    MatrixX3d f(250, 3);
    f.col(0) = VectorXd::LinSpaced(250, -60, 60);

    for (auto row : f.rowwise())
    {
        double k = calculateRationalQuadraticKernel(row[0], row[0], kernelParameters) + inv_beta;
        VectorXd kernel = calculateRationalQuadraticKernel(data.col(0), row[0], kernelParameters);

        Eigen::Matrix<double, 1, Eigen::Dynamic> common = kernel.transpose() * inv_covariance;
        // row: [mean, f, variance]
        row[1] = (common * data.col(1)).value();
        row[2] = k - (common * kernel).value();
    }

    return f;
}

void modelData(const MatrixX2d &data, double beta)
{
    // variance, alpha, length scale
    Vector3d kernelParameters = Vector3d::Constant(1);
    // calculate C from p(y) = N(y|0, C)
    MatrixXd covariance = calculateCovariance(data, beta, kernelParameters);

    // Optimization
    optim::algo_settings_t settings;
    // settings.gd_settings.par_step_size = 1e-4;
    settings.conv_failure_switch = 1;
    settings.vals_bound = true;
    settings.upper_bounds = Vector3d::Constant(1e5);
    settings.lower_bounds = Vector3d::Constant(1e-5);
    settings.print_level = 1;

    // externel data used in loss term
    LossArguments args{data, beta};
    VectorXd optimizedKernelParameters = kernelParameters;
    // optimize the kernel parameters by bfgs algorithm
    optim::bfgs(optimizedKernelParameters, evaluateOptimFn, reinterpret_cast<void*>(&args), settings);

    // calculate C after optimizing the kernel parameters
    MatrixXd optimizedCovariance = calculateCovariance(data, beta, optimizedKernelParameters);

    // sample data points from predictive distribution p(y*|y) with different covariance
    MatrixX3d f = generatePointsOfLine(data, beta, covariance, kernelParameters);
    MatrixX3d optimizedF = generatePointsOfLine(data, beta, optimizedCovariance, optimizedKernelParameters);

    std::cout << "Optimized kernel parameters (variance, alpha, length scale)" << std::endl;
    std::cout << optimizedKernelParameters << std::endl;

    // Plot the results
    showGUI(data, f, optimizedF);
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
