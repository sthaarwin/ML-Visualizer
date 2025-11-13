#pragma once
#include <vector>
#include "dataset.h"

struct LogisticModel {
    // Multiclass softmax weights: 3 classes x (bias + x + y)
    float W[3][3]; // W[c][0]=bias, W[c][1]=wx, W[c][2]=wy
    float lr;
    int epochs_trained;
    float last_loss;

    LogisticModel(float learning_rate = 0.5f);
    void randomize();
    // return vector of class probabilities (size 3)
    std::vector<float> predict_probs(float x, float y) const;
    int predict_label(float x, float y) const;
    float compute_loss(const std::vector<point2D>& data) const;
    void train_epoch(const std::vector<point2D>& data);
};
