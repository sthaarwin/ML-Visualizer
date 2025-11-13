#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#define IMGUI_IMPL_OPENGL_LOADER_GLEW

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>

#include "renderer.h"
#include "dataset.h"
#include "model.h"
#include <fstream>
#include <cmath>

std::vector<point2D> irisData;
std::vector<Vertex> irisVertices;
std::vector<Vertex> axisVertices;



int main() {

    std::cout << "Loading dataset..." << std::endl;
    irisData = LoadIrisDataset("../dataset/iris.csv");
    std::cout << "Loaded " << irisData.size() << " data points" << std::endl;
    
    irisVertices = irisToVertex(irisData);
    axisVertices = axesVertex();
    // Initialize logistic model (Setosa vs others)
    LogisticModel model(0.8f);
    model.randomize();
    
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

    // Initialize background grid and loss plot
    const int GRID_COLS = 80;
    const int GRID_ROWS = 80;
    initBackgroundGrid(GRID_COLS, GRID_ROWS);
    initLossPlot(512);

    std::vector<float> lossHistory;
    lossHistory.reserve(512);

    // Initial boundary update (3 pairwise lines -> 6 vertices)
    std::vector<Vertex> initialLines;
    initialLines.reserve(6);
    for(int i=0;i<3;++i){
        initialLines.push_back({-1.0f, 0.0f, 1,1,0});
        initialLines.push_back({ 1.0f, 0.0f, 1,1,0});
    }
    updateBoundaryLines(initialLines);

    while (!glfwWindowShouldClose(window)){
        // Poll events first
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Train one epoch per frame and update visualization
        model.train_epoch(irisData);

        // Append loss to history
        lossHistory.push_back(model.last_loss);
        if(lossHistory.size() > 512) lossHistory.erase(lossHistory.begin());

        // Update background confidence grid: compute probs for a regular grid
        std::vector<Vertex> bg; bg.reserve(GRID_COLS * GRID_ROWS);
        for(int r=0;r<GRID_ROWS;++r){
            for(int c=0;c<GRID_COLS;++c){
                float nx = (float)c / (GRID_COLS-1) * 2.0f - 1.0f; // -1..1
                float ny = (float)r / (GRID_ROWS-1) * 2.0f - 1.0f; // -1..1
                auto probs = model.predict_probs(nx, ny);
                // color blend by probability weighted sum of class colors
                float cr = probs[0]*0.0f + probs[1]*0.0f + probs[2]*1.0f; // blue, green, red ordering -> adjust
                float cg = probs[0]*0.0f + probs[1]*1.0f + probs[2]*0.0f;
                float cb = probs[0]*1.0f + probs[1]*0.0f + probs[2]*0.0f;
                // swap mapping to match class colors used for points: class0 blue, class1 green, class2 red
                Vertex v = { nx, ny, cr, cg, cb };
                bg.push_back(v);
            }
        }
        updateBackgroundGrid(bg);

        // Rebuild vertex array colored by multiclass predicted label
        for(size_t i=0;i<irisData.size();++i){
            Vertex v;
            v.x = irisData[i].x; v.y = irisData[i].y;
            int pred = model.predict_label(irisData[i].x, irisData[i].y);
            if(pred == 0){ v.r = 0.0f; v.g = 0.0f; v.b = 1.0f; }     // class 0 -> Blue
            else if(pred == 1){ v.r = 0.0f; v.g = 1.0f; v.b = 0.0f; } // class 1 -> Green
            else { v.r = 1.0f; v.g = 0.0f; v.b = 0.0f; }             // class 2 -> Red
            irisVertices[i] = v;
        }
        updateVertices(irisVertices);
        // Update decision boundary lines for each pair of classes (i,j)
        std::vector<Vertex> lines; lines.reserve(6);
        auto clamp = [](float v, float a, float b){ return v < a ? a : (v > b ? b : v); };
        for(int i=0;i<3;++i){
            for(int j=i+1;j<3;++j){
                // Solve (Wi - Wj) dot [1, x, y] = 0 -> y = - (delta_b + delta_wx * x) / delta_wy
                float db = model.W[i][0] - model.W[j][0];
                float dwx = model.W[i][1] - model.W[j][1];
                float dwy = model.W[i][2] - model.W[j][2];
                float xA = -1.0f, xB = 1.0f;
                float yA = 10.0f, yB = 10.0f;
                if(fabs(dwy) > 1e-6f){
                    yA = -(db + dwx * xA) / dwy;
                    yB = -(db + dwx * xB) / dwy;
                }
                yA = clamp(yA, -3.0f, 3.0f);
                yB = clamp(yB, -3.0f, 3.0f);
                // coloring: use yellow for boundaries
                Vertex va = {xA, yA, 1.0f, 1.0f, 0.0f};
                Vertex vb = {xB, yB, 1.0f, 1.0f, 0.0f};
                lines.push_back(va);
                lines.push_back(vb);
            }
        }
        updateBoundaryLines(lines);

        // Build UI
        ImGui::Begin("Controls");
        ImGui::Text("Epoch: %d", model.epochs_trained);
        ImGui::Text("Loss: %.4f", model.last_loss);
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
        // Draw background confidence first (subtle)
        drawBackgroundGrid();
        drawPoints(irisVertices.size());
        drawLines(axisVertices.size());
        // Draw decision boundaries
        drawBoundary();
        // Draw loss plot (top-right small panel)
        // Build loss plot vertices in NDC space (right side)
        int maxPts = 512;
        int n = (int)lossHistory.size();
        float x0 = 0.6f, x1 = 0.98f;
        float y0 = -0.95f, y1 = -0.6f;
        float maxLoss = 0.0f; for(float v: lossHistory) if(v>maxLoss) maxLoss = v;
        maxLoss = std::max(maxLoss, 1e-3f);
        std::vector<Vertex> lossVerts; lossVerts.reserve(n);
        for(int i=0;i<n;++i){
            float t = n==1?0.0f: (float)i / (float)(n-1);
            float x = x0 + t * (x1 - x0);
            float y = y0 + (lossHistory[i] / maxLoss) * (y1 - y0);
            // color white
            lossVerts.push_back({x, y, 1.0f, 1.0f, 1.0f});
        }
        updateLossPlot(lossVerts);
        drawLossPlot();

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
