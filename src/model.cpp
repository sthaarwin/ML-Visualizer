#include "model.h"
#include <random>
#include <cmath>
#include <fstream>

static void softmax_inplace(float logits[3], float probs[3]){
    // subtract max for numerical stability
    float m = logits[0];
    if(logits[1] > m) m = logits[1];
    if(logits[2] > m) m = logits[2];
    float sum = 0.0f;
    for(int i=0;i<3;++i){
        probs[i] = std::exp(logits[i] - m);
        sum += probs[i];
    }
    for(int i=0;i<3;++i) probs[i] /= sum;
}

LogisticModel::LogisticModel(float learning_rate)
    : lr(learning_rate), epochs_trained(0), last_loss(0.0f)
{
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) W[i][j] = 0.0f;
}

void LogisticModel::randomize(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(int i=0;i<3;++i){
        W[i][0] = dist(gen) * 0.5f; // bias smaller
        W[i][1] = dist(gen);
        W[i][2] = dist(gen);
    }
}

std::vector<float> LogisticModel::predict_probs(float x, float y) const{
    float logits[3];
    for(int c=0;c<3;++c) logits[c] = W[c][0] + W[c][1]*x + W[c][2]*y;
    float probs[3];
    softmax_inplace(logits, probs);
    return std::vector<float>(probs, probs+3);
}

int LogisticModel::predict_label(float x, float y) const{
    auto p = predict_probs(x,y);
    int best = 0;
    for(int i=1;i<3;++i) if(p[i] > p[best]) best = i;
    return best;
}

float LogisticModel::compute_loss(const std::vector<point2D>& data) const{
    if(data.empty()) return 0.0f;
    double loss = 0.0;
    for(const auto& p : data){
        int t = p.label; // 0,1,2
        float logits[3];
        for(int c=0;c<3;++c) logits[c] = W[c][0] + W[c][1]*p.x + W[c][2]*p.y;
        float probs[3]; softmax_inplace(logits, probs);
        float eps = 1e-7f;
        float pr = std::min(1.0f - eps, std::max(eps, probs[t]));
        loss += -std::log(pr);
    }
    return (float)(loss / data.size());
}

void LogisticModel::train_epoch(const std::vector<point2D>& data){
    if(data.empty()) return;
    // gradients dL/dW[c][k]
    double grad[3][3];
    for(int c=0;c<3;++c) for(int k=0;k<3;++k) grad[c][k] = 0.0;

    for(const auto& p : data){
        float logits[3];
        for(int c=0;c<3;++c) logits[c] = W[c][0] + W[c][1]*p.x + W[c][2]*p.y;
        float probs[3]; softmax_inplace(logits, probs);
        for(int c=0;c<3;++c){
            int y_true = (p.label == c) ? 1 : 0;
            float err = probs[c] - y_true; // derivative wrt logits
            grad[c][0] += err * 1.0f; // bias
            grad[c][1] += err * p.x;
            grad[c][2] += err * p.y;
        }
    }

    float invN = 1.0f / (float)data.size();
    for(int c=0;c<3;++c){
        for(int k=0;k<3;++k){
            float g = (float)(grad[c][k] * invN);
            W[c][k] -= lr * g;
        }
    }

    epochs_trained += 1;
    last_loss = compute_loss(data);
}

bool LogisticModel::save(const char* filename) const{
    std::ofstream out(filename, std::ios::binary);
    if(!out) return false;
    out.write((const char*)W, sizeof(W));
    out.write((const char*)&lr, sizeof(lr));
    out.write((const char*)&epochs_trained, sizeof(epochs_trained));
    out.write((const char*)&last_loss, sizeof(last_loss));
    out.close();
    return true;
}

bool LogisticModel::load(const char* filename){
    std::ifstream in(filename, std::ios::binary);
    if(!in) return false;
    in.read((char*)W, sizeof(W));
    in.read((char*)&lr, sizeof(lr));
    in.read((char*)&epochs_trained, sizeof(epochs_trained));
    in.read((char*)&last_loss, sizeof(last_loss));
    in.close();
    return true;
}
