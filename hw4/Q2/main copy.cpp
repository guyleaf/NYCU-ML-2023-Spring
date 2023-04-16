#include <random>
#include <string>
#include <iostream>
#include <array>
#include <vector>
#include <variant>
#include <iterator>

#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#define EIGEN_USE_MKL_ALL

#include <Eigen/Core>
#include <Eigen/LU>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

const char TITLE[] = "ML HW4";
const std::string TRAIN_IMAGES_FILE = "train-images.idx3-ubyte";
const std::string TRAIN_LABELS_FILE = "train-labels.idx1-ubyte";

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
        auto uniform = [&uniform_dist, &rng = this->rng]()
        { return uniform_dist(rng); };
        return Eigen::MatrixXd::NullaryExpr(n, m, uniform);
    }
};

// for MNIST images
constexpr int IMAGE_DATA_ID = 2051;
constexpr int LABEL_DATA_ID = 2049;

using MatrixX10d = Eigen::Matrix<double, Eigen::Dynamic, 10>;
using Matrix10Xd = Eigen::Matrix<double, 10, Eigen::Dynamic>;
using MatrixXb = Eigen::Matrix<std::byte, Eigen::Dynamic, Eigen::Dynamic>;
using Vector10d = Eigen::Vector<double, 10>;
using VectorXb = Eigen::VectorX<std::byte>;
using STLArray4b = std::array<std::byte, 4>;

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
};

struct ImagesFileBody : FileBody
{
    ImageSize imageSize;
    Eigen::MatrixXi images;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LabelsFileBody : FileBody
{
    Eigen::VectorXi labels;
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

LabelsFileBody parseMNISTLabelsFile(const fs::path &path)
{
    std::ifstream ifs(path.generic_string(), std::ios::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    STLArray4b bytes;
    auto sizeOfFrame = bytes.size();

    ifs.read(std::bit_cast<char *>(std::bit_cast<char *>(bytes.data())), sizeOfFrame);
    int magicNumber = castToInt(bytes, true);

    ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
    int numberOfItems = castToInt(bytes, true);

    if (magicNumber != LABEL_DATA_ID)
    {
        throw std::runtime_error("The file is not in label format.");
    }

    std::vector<std::byte> buffer(numberOfItems);
    ifs.read(std::bit_cast<char *>(buffer.data()), numberOfItems);

    ifs.close();

    Eigen::Map<VectorXb> tmp = VectorXb::Map(buffer.data(), numberOfItems, 1);
    return {magicNumber, numberOfItems, tmp.cast<int>()};
}

ImagesFileBody parseMNISTImagesFile(const fs::path &path)
{
    std::ifstream ifs(path.generic_string(), std::ios::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open the file.");
    }

    STLArray4b bytes;
    auto sizeOfFrame = bytes.size();

    ifs.read(std::bit_cast<char *>(std::bit_cast<char *>(bytes.data())), sizeOfFrame);
    int magicNumber = castToInt(bytes, true);

    ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
    int numberOfItems = castToInt(bytes, true);

    if (magicNumber != IMAGE_DATA_ID)
    {
        throw std::runtime_error("The file is not in label format.");
    }

    ImageSize imageSize;
    ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
    imageSize.height = castToInt(bytes, true);

    ifs.read(std::bit_cast<char *>(bytes.data()), sizeOfFrame);
    imageSize.width = castToInt(bytes, true);

    int sizePerImage = imageSize.size();
    int totalPixelsOrLabels = numberOfItems * sizePerImage;

    std::vector<std::byte> buffer(totalPixelsOrLabels);
    ifs.read(std::bit_cast<char *>(buffer.data()), totalPixelsOrLabels);

    ifs.close();

    Eigen::Map<MatrixXb> tmp = MatrixXb::Map(buffer.data(), numberOfItems, sizePerImage);
    return {magicNumber, numberOfItems, imageSize, tmp.cast<int>()};
}

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

std::pair<Eigen::MatrixXi, Eigen::MatrixXi> preprocess(const ImagesFileBody &imagesFile, const LabelsFileBody &labelsFile)
{
    Eigen::MatrixXi images = imagesFile.images.cast<int>();
    Eigen::VectorXi labels = labelsFile.labels.cast<int>();
    std::cout << (images.array() >= 128).count() << std::endl;
    return std::make_pair((images.array() >= 128).cast<int>().matrix(), labels);
}

#pragma endregion

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

    // parse MNIST files
    auto trainImages = parseMNISTImagesFile(path / TRAIN_IMAGES_FILE);
    auto trainLabels = parseMNISTLabelsFile(path / TRAIN_LABELS_FILE);

    auto [images, labels] = preprocess(trainImages, trainLabels);

    return 0;
}
