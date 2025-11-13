//renderer.h
#pragma once

#include <vector>
#include "dataset.h"

struct Vertex{
    float x, y; //normalized data
    float r, g, b; //color
};

extern int windowHeight;
extern int windowWidth;

//Modern OpenGL Functions
void initRenderer(const std::vector<Vertex>& pointVertices, const std::vector<Vertex>& axisVertices);
void drawPoints(size_t numPoints);
void drawLines(size_t numVertices);
void updateVertices(const std::vector<Vertex>& vertices);
void setPointVertices(const std::vector<Vertex>& vertices); // reallocate point VBO for different dataset sizes
// Test points (user-provided)
void initTestPoints(int maxPoints);
void updateTestPoints(const std::vector<Vertex>& testVertices);
void drawTestPoints();

// Decision boundary support (up to three pairwise lines -> 6 vertices total)
void updateBoundaryLines(const std::vector<Vertex>& lineVertices); // expect 6 vertices (3 lines)
void drawBoundary();
// Background/confidence grid
void initBackgroundGrid(int cols, int rows);
void updateBackgroundGrid(const std::vector<Vertex>& gridVertices);
void drawBackgroundGrid();

// Loss plot
void initLossPlot(int maxPoints);
void updateLossPlot(const std::vector<Vertex>& plotVertices);
void drawLossPlot();

//Manual Drawing
void drawCircleManual(int xc, int yc, int radius, float r, float g, float b);
void drawLineManual(int x0, int y0, int x1, int y1, int r, int g, int b);


//helper function
std::vector<Vertex> irisToVertex(const std::vector<point2D>& data);
std::vector<Vertex> axesVertex();

//map normalized to [-1, 1] coordinates to pixel coordinates
int mapX(float xNorm, int width=800);
int mapY(float yNorm, int height=600);

void drawIrisPoints(const std::vector<point2D> data, int radius);

void drawAxes();