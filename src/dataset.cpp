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
    auto trim = [](std::string &s){
        // remove leading/trailing spaces and CR
        while(!s.empty() && (s.back()=='\r' || s.back()=='\n' || s.back()==' ' || s.back()=='\t')) s.pop_back();
        size_t i=0; while(i<s.size() && (s[i]==' '||s[i]=='\t')) ++i;
        if(i>0) s = s.substr(i);
    };

    while(getline(file, line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string val;

        float sepalLength=0.0f, sepalWidth=0.0f, petalLength=0.0f, petalWidth=0.0f;
        std::string variety;

        try{
            if(!std::getline(ss, val, ',')) { std::cerr<<"Skipping malformed CSV line (missing field): "<<line<<"\n"; continue; }
            trim(val);
            sepalLength = std::stof(val);

            if(!std::getline(ss, val, ',')) { std::cerr<<"Skipping malformed CSV line (missing field): "<<line<<"\n"; continue; }
            trim(val);
            sepalWidth = std::stof(val);

            if(!std::getline(ss, val, ',')) { std::cerr<<"Skipping malformed CSV line (missing field): "<<line<<"\n"; continue; }
            trim(val);
            petalLength = std::stof(val);

            if(!std::getline(ss, val, ',')) { std::cerr<<"Skipping malformed CSV line (missing field): "<<line<<"\n"; continue; }
            trim(val);
            petalWidth = std::stof(val);

            if(!std::getline(ss, val, ',')) { // last field may be without trailing comma
                // try to take remaining
                if(!std::getline(ss, val)) val = "";
            }
            trim(val);
            variety = val;
            if(!variety.empty() && variety.front() == '"' && variety.back() == '"') variety = variety.substr(1, variety.size()-2);

        } catch(const std::exception &e){
            std::cerr << "Failed to parse line: '"<< line <<"' -> "<< e.what() <<"\n";
            continue;
        }

        int label = (variety == "Setosa") ? 0 : (variety == "Versicolor") ? 1 : 2;

        // Normalize petal features to [-1,1]
        float x = (petalLength - 1.0f) / (6.9f - 1.0f) * 2.0f - 1.0f;
        float y = (petalWidth  - 0.1f) / (2.5f - 0.1f) * 2.0f - 1.0f;

        data.push_back({x, y, label});
    }
    return data;
}