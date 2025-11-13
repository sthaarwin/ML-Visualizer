#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#define IMGUI_IMPL_OPENGL_LOADER_GLEW

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>

#include "renderer.h"
#include "dataset.h"
#include <fstream>

std::vector<point2D> irisData;
std::vector<Vertex> irisVertices;
std::vector<Vertex> axisVertices;



int main() {

    std::cout << "Loading dataset..." << std::endl;
    irisData = LoadIrisDataset("../dataset/iris.csv");
    std::cout << "Loaded " << irisData.size() << " data points" << std::endl;
    
    irisVertices = irisToVertex(irisData);
    axisVertices = axesVertex();
    
    // Initialize GLFW
    if (!glfwInit()){
        std::cout <<"glfw init failed"<<std::endl;
        return -1;
    }

    // Create window and OpenGL context first
    GLFWwindow* window = glfwCreateWindow(800, 600, "ML Visualizer", NULL, NULL);
    if (!window) { std::cout << "glfwCreateWindow FAILED" << std::endl; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    // Initialize GLEW after context is current
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cout<<"glew init failed"<<std::endl;
        return -1;
    }

    // Setup ImGui (after GL context + GLEW)
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Now initialize our GL resources (VAO/VBO etc.)
    initRenderer(irisVertices, axisVertices);

    while (!glfwWindowShouldClose(window)){
        // Poll events first
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Build UI
        ImGui::Begin("Controls");
        ImGui::Text("ML Visualizer Ready");
        ImGui::End();

        // Render UI to get draw data
        ImGui::Render();

        // Rendering
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // GPU drawing
        drawPoints(irisVertices.size());
        drawLines(axisVertices.size());

        // Manual immediate-mode drawing removed (using modern GL only)

        // Render ImGui on top
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers once
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
