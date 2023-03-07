#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <algebra/algebra.hpp>
#include <algebra/matrix.hpp>

#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#define TITLE "ML HW1"

algebra::Matrix2d<double> getPointsFromFunction(const algebra::Matrix2d<double> &weights, double start = -6, double end = 6, int count = 200);
algebra::Matrix2d<double> generateDesignMatrix(const algebra::Matrix2d<double> &points, int n);

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
    GLFWwindow *window = glfwCreateWindow(1920, 1080, TITLE, NULL, NULL);
    if (window == NULL)
        return nullptr;

    glfwMakeContextCurrent(window);
    // glfwSwapInterval(1); // Enable vsync
    glfwSwapInterval(0);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    // (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    float SCALE = 1.2f;
    ImFontConfig cfg;
    cfg.SizePixels = 15 * SCALE;
    ImFont *font = io.Fonts->AddFontDefault(&cfg);

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    // - Our Emscripten build process allows embedding fonts to be accessible at runtime from the "fonts/" folder. See Makefile.emscripten for details.
    // io.Fonts->AddFontDefault();
    // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    // ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    // IM_ASSERT(font != NULL);

    return window;
}

void showGUI(const algebra::Matrix2d<double> &points, algebra::Matrix2d<double> &weightsForRLSE, algebra::Matrix2d<double> &weightsForNewton)
{
    auto window = setUpGUI();
    if (window == nullptr)
    {
        throw std::runtime_error("Cannot create window.");
    }

    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImVec4 point_color = ImVec4(255, 0, 0, 1);

    const auto pointsForRLSE = getPointsFromFunction(weightsForRLSE);
    const auto pointsForNewton = getPointsFromFunction(weightsForNewton);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            const int limits = 10;
            const auto windowSize = ImGui::GetIO().DisplaySize;

            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(windowSize.x * 0.667, windowSize.y), ImGuiCond_Always);
            ImGui::Begin("Result", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDecoration);

            ImGui::Text("Result");
            ImGui::Spacing();

            if (ImPlot::BeginPlot("rLSE"))
            {
                ImPlot::SetupAxes("x", "y");
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -limits, limits);

                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, point_color, IMPLOT_AUTO, point_color);
                ImPlot::PlotScatter("data", &points.col(0)[0], &points.col(1)[0], points.rows());

                ImPlot::SetNextLineStyle(ImPlot::SampleColormap(0.2, ImPlotColormap_Jet));
                ImPlot::PlotLine("f(x)", &pointsForRLSE.col(0)[0], &pointsForRLSE.col(1)[0], pointsForRLSE.rows());
                ImPlot::EndPlot();
            }

            if (ImPlot::BeginPlot("Newton's method"))
            {
                ImPlot::SetupAxes("x", "y");
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -limits, limits);

                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, IMPLOT_AUTO, point_color, IMPLOT_AUTO, point_color);
                ImPlot::PlotScatter("data", &points.col(0)[0], &points.col(1)[0], points.rows());

                ImPlot::SetNextLineStyle(ImPlot::SampleColormap(0.8, ImPlotColormap_Jet));
                ImPlot::PlotLine("f(x)", &pointsForNewton.col(0)[0], &pointsForNewton.col(1)[0], pointsForNewton.rows());
                ImPlot::EndPlot();
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

algebra::Matrix2d<double> getPointsFromFunction(const algebra::Matrix2d<double> &weights, double start, double end, int count)
{
    algebra::Matrix2d<double> points(count, 2);
    auto n = weights.rows();
    double scale = (end - start) / static_cast<double>(count);
    for (int i = 0; i < count; i++)
    {
        points(i, 0) = start;

        auto designMatrix = generateDesignMatrix(algebra::Matrix2d<double>(1, 1, start), n);
        points(i, 1) = (designMatrix * weights).sum();

        start += scale;
    }

    return points;
}

algebra::Matrix2d<double> generateDesignMatrix(const algebra::Matrix2d<double> &points, int n)
{
    algebra::Matrix2d<double> matrix(points.rows(), n, 1);
    auto xs = points.col(0);
    for (int i = 1; i < n; i++)
    {
        matrix.col(i) = xs;
        xs *= xs;
    }
    return matrix;
}

algebra::Matrix2d<double> fitPointsByRLSE(const algebra::Matrix2d<double> &a, const algebra::Matrix2d<double> &b, double lambda)
{
    auto transposedA = a.transpose();
    std::size_t n = a.cols();
    return (transposedA * a + lambda * algebra::eye<double>(n)).inverse() * transposedA * b;
}

algebra::Matrix2d<double> fitPointsByLSEWithNewton(const algebra::Matrix2d<double> &a, const algebra::Matrix2d<double> &b)
{
    auto transposedA = a.transpose();
    return (transposedA * a).inverse() * transposedA * b;
}

algebra::Matrix2d<double> parseInput()
{
    std::vector<double> points;

    std::string::size_type i;
    double tmp;
    while (!std::cin.eof())
    {
        std::cin >> tmp;
        points.push_back(tmp);

        // ignore comma
        std::cin.ignore(1);

        std::cin >> tmp;
        points.push_back(tmp);
    }

    std::cout << "Finished parsing input" << std::endl;

    return algebra::Matrix2d<double>(points.size() / 2, points);
}

double calculateLSE(const algebra::Matrix2d<double> &a, const algebra::Matrix2d<double> &b, const algebra::Matrix2d<double> &weights)
{
    return (a * weights - b).pow(2).sum();
}

std::string describeLine(const algebra::Matrix2d<double> &weights)
{
    std::ostringstream oss;
    oss.precision(12);
    oss << std::fixed;
    auto n = weights.rows();
    for (auto i = weights.rows(); i > 0; i--)
    {
        auto value = weights(i - 1, 0);
        if (value >= 0 && i != n)
        {
            oss << " + ";
        }
        else if (value < 0)
        {
            oss << " - ";
        }
        oss << std::abs(value);
        if (i > 1)
        {
            oss << "x^" << i - 1;
        }
    }
    return oss.str();
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "The arguments must include two values, bases, lambda." << std::endl;
        return 1;
    }

    std::cout.precision(10);
    std::cout << std::fixed;

    int n = std::stoi(argv[1]);
    double lambda = std::stoi(argv[2]);

    auto points = parseInput();
    auto a = generateDesignMatrix(points, n);
    auto b = algebra::Matrix2d<double>(points.rows(), 1, points.col(1));

    auto weightsForRLSE = fitPointsByRLSE(a, b, lambda);
    std::cout << "LSE:" << std::endl;
    std::cout << "Fitting line: " << describeLine(weightsForRLSE) << std::endl;
    std::cout << "Total error: " << calculateLSE(a, b, weightsForRLSE) << std::endl;

    std::cout << std::endl;

    auto weightsForNewton = fitPointsByLSEWithNewton(a, b);
    std::cout << "Newton's Method:" << std::endl;
    std::cout << "Fitting line: " << describeLine(weightsForNewton) << std::endl;
    std::cout << "Total error: " << calculateLSE(a, b, weightsForNewton) << std::endl;

    showGUI(points, weightsForRLSE, weightsForNewton);
    return 0;
}