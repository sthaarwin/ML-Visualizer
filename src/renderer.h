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