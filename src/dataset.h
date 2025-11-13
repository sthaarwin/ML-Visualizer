//dataset.h

#pragma once
#include<vector>

struct point2D{
    float x, y;
    int label;
};

std::vector<point2D> LoadIrisDataset(const char* filename);