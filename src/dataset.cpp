//datset.cpp

#include "dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

std::vector<point2D> LoadIrisDataset(const char* filename){
    std::vector<point2D> data;
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cout <<" Failed to open CSV"<<std::endl;
        return data;
    }

    std::string line;
    getline(file, line); //skipping the header

    while(getline(file, line)){
        std::stringstream ss(line);
        std::string val;

        float sepalLength, sepalWidth, petalLength, petalWidth;
        std::string variety;

        getline (ss, val, ',');sepalLength = std::stof(val);
        getline (ss, val, ',');sepalWidth = std::stof(val);
        getline (ss, val, ',');petalLength = std::stof(val);
        getline (ss, val, ',');petalWidth = std::stof(val);
        getline (ss, val, ',');variety = val;

        if (variety[0] == '"') variety = variety.substr(1, variety.size()-2); // remove quotes
        
       int label = (variety == "Setosa") ? 0 : (variety == "Versicolor") ? 1 : 2;

        // Normalize petal features to [-1,1]
        float x = (petalLength - 1.0f) / (6.9f - 1.0f) * 2.0f - 1.0f;
        float y = (petalWidth  - 0.1f) / (2.5f - 0.1f) * 2.0f - 1.0f;

        data.push_back({x, y, label});
    }
    return data;
}