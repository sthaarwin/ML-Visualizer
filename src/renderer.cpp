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

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    // Debug info
    printf("initRenderer: shaderProgram=%u, pointsVAO=%u, axesVAO=%u\n", shaderProgram, VAO_points, VAO_axes);
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