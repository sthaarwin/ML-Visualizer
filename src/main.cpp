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
std::vector<Vertex> testVertices;


int main() {

    std::cout << "Loading dataset..." << std::endl;
    irisData = LoadIrisDataset("../dataset/synthetic.csv");
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
    // initialize test point buffer (small fixed capacity)
    initTestPoints(64);

    // Dataset selector state
    const char* datasetFiles[] = { "iris.csv", "synthetic.csv", "synthetic_nonlinear.csv" };
    static int datasetIndex = 1; // default to synthetic
    static bool randomizeOnLoad = true;

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

        // Build UI (controls)
        static bool paused = false;
        static int epochsPerFrame = 1;
        static float lr_ui = model.lr;

        ImGui::Begin("Controls");
        // Dataset selector
        if(ImGui::Combo("Dataset", &datasetIndex, datasetFiles, IM_ARRAYSIZE(datasetFiles))){
            // user changed selection; reload immediately
            std::string path = std::string("../dataset/") + datasetFiles[datasetIndex];
            auto newData = LoadIrisDataset(path.c_str());
            if(!newData.empty()){
                irisData = newData;
                irisVertices = irisToVertex(irisData);
                setPointVertices(irisVertices); // reallocate VBO for new size
                if(randomizeOnLoad) model.randomize();
                lossHistory.clear();
            }
        }
        ImGui::SameLine();
        ImGui::Checkbox("Randomize on Load", &randomizeOnLoad);
        ImGui::Text("Epoch: %d", model.epochs_trained);
        ImGui::Text("Loss: %.4f", model.last_loss);
        if(ImGui::SliderFloat("Learning Rate", &lr_ui, 0.0001f, 2.0f, "%.4f")){
            model.lr = lr_ui;
        }
        ImGui::SliderInt("Epochs / Frame", &epochsPerFrame, 0, 10);
        if(ImGui::Button(paused ? "Resume" : "Pause")) paused = !paused;
        ImGui::SameLine();
        if(ImGui::Button("Randomize Model")) model.randomize();
        if(ImGui::Button("Save Model")) model.save("model.bin");
        ImGui::SameLine();
        if(ImGui::Button("Load Model")){
            if(model.load("model.bin")) lr_ui = model.lr;
        }
        ImGui::End();

        // Test point UI
        ImGui::Begin("Test Point");
        static float test_x = 0.0f, test_y = 0.0f;
        ImGui::SliderFloat("X (-1..1)", &test_x, -1.0f, 1.0f);
        ImGui::SliderFloat("Y (-1..1)", &test_y, -1.0f, 1.0f);
        if(ImGui::Button("Predict")){
            auto p = model.predict_probs(test_x, test_y);
            ImGui::Text("Prob class0 (blue): %.3f", p[0]);
            ImGui::Text("Prob class1 (green): %.3f", p[1]);
            ImGui::Text("Prob class2 (red): %.3f", p[2]);
        }
        ImGui::SameLine();
        if(ImGui::Button("Add To Plot")){
            // add a test vertex colored by predicted label
            auto p = model.predict_probs(test_x, test_y);
            int pred = model.predict_label(test_x, test_y);
            Vertex v;
            v.x = test_x; v.y = test_y;
            if(pred == 0){ v.r = 0.0f; v.g = 0.0f; v.b = 1.0f; }
            else if(pred == 1){ v.r = 0.0f; v.g = 1.0f; v.b = 0.0f; }
            else { v.r = 1.0f; v.g = 0.0f; v.b = 0.0f; }
            testVertices.push_back(v);
            // limit
            if(testVertices.size() > 64) testVertices.erase(testVertices.begin());
            updateTestPoints(testVertices);
        }
        ImGui::End();

        // Training step(s)
        if(!paused && epochsPerFrame > 0){
            for(int e=0;e<epochsPerFrame;++e){
                model.train_epoch(irisData);
                // Append loss to history per epoch
                lossHistory.push_back(model.last_loss);
                if(lossHistory.size() > 512) lossHistory.erase(lossHistory.begin());
            }
        }

        // Update background confidence grid: compute probs for a regular grid
        std::vector<Vertex> bg; bg.reserve(GRID_COLS * GRID_ROWS);
        for(int r=0;r<GRID_ROWS;++r){
            for(int c=0;c<GRID_COLS;++c){
                float nx = (float)c / (GRID_COLS-1) * 2.0f - 1.0f; // -1..1
                float ny = (float)r / (GRID_ROWS-1) * 2.0f - 1.0f; // -1..1
                auto probs = model.predict_probs(nx, ny);
                // color blend by probability weighted sum of class colors
                float cr = probs[0]*0.0f + probs[1]*0.0f + probs[2]*1.0f; // class2 red
                float cg = probs[0]*0.0f + probs[1]*1.0f + probs[2]*0.0f; // class1 green
                float cb = probs[0]*1.0f + probs[1]*0.0f + probs[2]*0.0f; // class0 blue
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
        // draw any user test points on top of dataset points
        drawTestPoints();
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
