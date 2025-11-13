//renderer.cpp

#include "renderer.h"
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>

//GPU rendering
unsigned int VAO_points = 0, VBO_points = 0;
unsigned int VAO_axes = 0, VBO_axes = 0;
unsigned int shaderProgram = 0;
unsigned int VAO_boundary = 0, VBO_boundary = 0;
unsigned int VAO_bg = 0, VBO_bg = 0;
unsigned int VAO_loss = 0, VBO_loss = 0;
unsigned int VAO_test = 0, VBO_test = 0;
int bg_cols = 0, bg_rows = 0;
int loss_point_count = 0;
int test_point_count = 0;

//global variable
int windowWidth = 800;
int windowHeight = 600;

int mapX(float xNorm, int width){
    return int((xNorm + 1.0f) * 0.5f * width);
}

int mapY(float yNorm, int height){
    return int((yNorm + 1.0f) * 0.5f * height);
}


// ------------------ Modern OpenGL ------------------
static GLuint compileShader(GLenum type, const char* src){
    GLuint id = glCreateShader(type);
    glShaderSource(id, 1, &src, NULL);
    glCompileShader(id);
    GLint ok = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if(!ok){
        char buf[1024]; buf[0]=0;
        glGetShaderInfoLog(id, sizeof(buf), NULL, buf);
        fprintf(stderr, "Shader compile error: %s\n", buf);
    }
    return id;
}

static GLuint createSimpleProgram(){
    const char* vs = "#version 330 core\n"
                     "layout(location = 0) in vec2 aPos;\n"
                     "layout(location = 1) in vec3 aColor;\n"
                     "out vec3 vColor;\n"
                     "void main(){ vColor = aColor; gl_Position = vec4(aPos, 0.0, 1.0); }\n";
    const char* fs = "#version 330 core\n"
                     "in vec3 vColor;\n"
                     "out vec4 FragColor;\n"
                     "void main(){ FragColor = vec4(vColor, 1.0); }\n";
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ char buf[1024]; buf[0]=0; glGetProgramInfoLog(p, sizeof(buf), NULL, buf); fprintf(stderr, "Program link error: %s\n", buf); }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

void initRenderer(const std::vector<Vertex>& pointVertices, const std::vector<Vertex>& axisVertices){
    // create shader program
    shaderProgram = createSimpleProgram();

    // Points VAO/VBO
    glGenVertexArrays(1, &VAO_points);
    glGenBuffers(1, &VBO_points);
    glBindVertexArray(VAO_points);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_points);
    glBufferData(GL_ARRAY_BUFFER, pointVertices.size()*sizeof(Vertex), pointVertices.data(), GL_DYNAMIC_DRAW);
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    // Color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Axes VAO/VBO
    glGenVertexArrays(1, &VAO_axes);
    glGenBuffers(1, &VBO_axes);
    glBindVertexArray(VAO_axes);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_axes);
    glBufferData(GL_ARRAY_BUFFER, axisVertices.size()*sizeof(Vertex), axisVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Boundary VAO/VBO (up to three pairwise lines -> 6 vertices)
    glGenVertexArrays(1, &VAO_boundary);
    glGenBuffers(1, &VBO_boundary);
    glBindVertexArray(VAO_boundary);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_boundary);
    Vertex boundaryInit[6] = {
        {-1.0f, 0.0f, 1,1,0}, {1.0f, 0.0f, 1,1,0},
        {-1.0f, 0.0f, 1,1,0}, {1.0f, 0.0f, 1,1,0},
        {-1.0f, 0.0f, 1,1,0}, {1.0f, 0.0f, 1,1,0}
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(boundaryInit), boundaryInit, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Background VAO/VBO (initialized on demand)
    VAO_bg = 0; VBO_bg = 0;
    // Loss plot VAO/VBO (initialized on demand)
    VAO_loss = 0; VBO_loss = 0;
    // Test points VAO/VBO (small fixed buffer)
    VAO_test = 0; VBO_test = 0;

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    // Debug info
    printf("initRenderer: shaderProgram=%u, pointsVAO=%u, axesVAO=%u\n", shaderProgram, VAO_points, VAO_axes);
}

void initBackgroundGrid(int cols, int rows){
    bg_cols = cols; bg_rows = rows;
    if(VAO_bg) { glDeleteVertexArrays(1, &VAO_bg); glDeleteBuffers(1, &VBO_bg); }
    glGenVertexArrays(1, &VAO_bg);
    glGenBuffers(1, &VBO_bg);
    glBindVertexArray(VAO_bg);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_bg);
    // allocate but empty for now
    glBufferData(GL_ARRAY_BUFFER, cols * rows * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void updateBackgroundGrid(const std::vector<Vertex>& gridVertices){
    if(!VAO_bg || !VBO_bg) return;
    glBindBuffer(GL_ARRAY_BUFFER, VBO_bg);
    glBufferSubData(GL_ARRAY_BUFFER, 0, gridVertices.size()*sizeof(Vertex), gridVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawBackgroundGrid(){
    if(!shaderProgram || !VAO_bg) return;
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO_bg);
    glPointSize(2.0f);
    glDrawArrays(GL_POINTS, 0, bg_cols * bg_rows);
    glBindVertexArray(0);
    glUseProgram(0);
}

void initLossPlot(int maxPoints){
    if(VAO_loss) { glDeleteVertexArrays(1, &VAO_loss); glDeleteBuffers(1, &VBO_loss); }
    glGenVertexArrays(1, &VAO_loss);
    glGenBuffers(1, &VBO_loss);
    glBindVertexArray(VAO_loss);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_loss);
    glBufferData(GL_ARRAY_BUFFER, maxPoints * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void updateLossPlot(const std::vector<Vertex>& plotVertices){
    if(!VAO_loss || !VBO_loss) return;
    glBindBuffer(GL_ARRAY_BUFFER, VBO_loss);
    glBufferSubData(GL_ARRAY_BUFFER, 0, plotVertices.size()*sizeof(Vertex), plotVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    loss_point_count = (int)plotVertices.size();
}

void drawLossPlot(){
    if(!shaderProgram || !VAO_loss) return;
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO_loss);
    glLineWidth(2.0f);
    if(loss_point_count > 0) glDrawArrays(GL_LINE_STRIP, 0, loss_point_count);
    glBindVertexArray(0);
    glUseProgram(0);
}

void drawPoints(size_t numPoints){
    if(!shaderProgram) return;
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO_points);
    glPointSize(6.0f);
    glDrawArrays(GL_POINTS, 0, (GLsizei)numPoints);
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) printf("glDrawArrays(GL_POINTS) error: 0x%04x\n", err);
    glBindVertexArray(0);
    glUseProgram(0);
}

void drawLines(size_t numVertices){
    if(!shaderProgram) return;
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO_axes);
    glDrawArrays(GL_LINES, 0, (GLsizei)numVertices);
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) printf("glDrawArrays(GL_LINES) error: 0x%04x\n", err);
    glBindVertexArray(0);
    glUseProgram(0);
}

void updateVertices(const std::vector<Vertex>& vertices){
    glBindBuffer(GL_ARRAY_BUFFER, VBO_points);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size()*sizeof(Vertex), vertices.data());
}

void setPointVertices(const std::vector<Vertex>& vertices){
    glBindVertexArray(VAO_points);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_points);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    // re-specify attributes in case driver state changed
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Convert Iris dataset to Vertex array
std::vector<Vertex> irisToVertex(const std::vector<point2D>& data){
    std::vector<Vertex> vertices;
    for(auto& p : data){
        Vertex v;
        v.x = p.x;
        v.y = p.y;
        if(p.label == 0){ v.r=0; v.g=0; v.b=1; }       // Blue
        else if(p.label==1){ v.r=0; v.g=1; v.b=0; }    // Green
        else{ v.r=1; v.g=0; v.b=0; }                   // Red
        vertices.push_back(v);
    }
    return vertices;
}

// Axes vertices
std::vector<Vertex> axesVertex(){
    std::vector<Vertex> axes;
    // X-axis from (-1,0) to (1,0)
    axes.push_back({-1.0f, 0.0f, 1,1,1});
    axes.push_back({ 1.0f, 0.0f, 1,1,1});
    // Y-axis from (0,-1) to (0,1)
    axes.push_back({0.0f, -1.0f, 1,1,1});
    axes.push_back({0.0f,  1.0f, 1,1,1});
    return axes;
}

// ------------------ Manual Algorithms ------------------
// Manual Midpoint Circle Algorithm
void drawCircleManual(int xc, int yc, int radius, float r, float g, float b){
    int x = 0;
    int y = radius;
    int d = 1 - radius;

    auto plotCirclePoints = [&](int px, int py){
        glBegin(GL_POINTS);
        glColor3f(r, g, b);
        glVertex2i(xc + px, yc + py);
        glVertex2i(xc - px, yc + py);
        glVertex2i(xc + px, yc - py);
        glVertex2i(xc - px, yc - py);
        glVertex2i(xc + py, yc + px);
        glVertex2i(xc - py, yc + px);
        glVertex2i(xc + py, yc - px);
        glVertex2i(xc - py, yc - px);
        glEnd();
    };

    plotCirclePoints(x, y);
    while(x < y){
        x++;
        if(d < 0) d += 2*x + 1;
        else { y--; d += 2*(x - y) + 1; }
        plotCirclePoints(x, y);
    }
}

// Manual Bresenham Line Algorithm
void drawLineManual(int x0, int y0, int x1, int y1, int r, int g, int b){
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while(true){
        glBegin(GL_POINTS);
        glColor3f((float)r, (float)g, (float)b);
        glVertex2i(x0, y0);
        glEnd();

        if(x0 == x1 && y0 == y1) break;
        e2 = 2*err;
        if(e2 >= dy){ err += dy; x0 += sx; }
        if(e2 <= dx){ err += dx; y0 += sy; }
    }
}

void updateBoundaryLines(const std::vector<Vertex>& lineVertices){
    // Expect up to 6 vertices (3 lines). If fewer provided, pad with off-screen values.
    Vertex tmp[6];
    size_t n = lineVertices.size();
    for(size_t i=0;i<6;++i){
        if(i < n) tmp[i] = lineVertices[i];
        else tmp[i] = {10.0f, 10.0f, 1,1,0};
    }
    glBindBuffer(GL_ARRAY_BUFFER, VBO_boundary);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(tmp), tmp);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawBoundary(){
    if(!shaderProgram) return;
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO_boundary);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
}

void initTestPoints(int maxPoints){
    if(VAO_test) { glDeleteVertexArrays(1, &VAO_test); glDeleteBuffers(1, &VBO_test); }
    glGenVertexArrays(1, &VAO_test);
    glGenBuffers(1, &VBO_test);
    glBindVertexArray(VAO_test);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_test);
    glBufferData(GL_ARRAY_BUFFER, maxPoints * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void updateTestPoints(const std::vector<Vertex>& testVertices){
    if(!VAO_test || !VBO_test) return;
    glBindBuffer(GL_ARRAY_BUFFER, VBO_test);
    glBufferSubData(GL_ARRAY_BUFFER, 0, testVertices.size()*sizeof(Vertex), testVertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    test_point_count = (int)testVertices.size();
}

void drawTestPoints(){
    if(!shaderProgram || !VAO_test) return;
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO_test);
    glPointSize(10.0f);
    if(test_point_count > 0) glDrawArrays(GL_POINTS, 0, test_point_count);
    glBindVertexArray(0);
    glUseProgram(0);
}