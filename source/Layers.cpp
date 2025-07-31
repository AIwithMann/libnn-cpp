#include "Layers.h"
#include <cmath>
#include <random>
#include<Eigen/Dense>
#include <stdexcept>

static std::random_device rd;
static std::mt19937 gen(rd());

void Layer::setTraining(bool mode){
    isTraining = mode;
}

Linear::Linear(int nInputs, int nOutputs, float low, float high){
    this->B = Eigen::MatrixXf::Zero(nOutputs,1);
    this->bGrad.resize(nOutputs,1);
    this->W.resize(nOutputs,nInputs);
    this->wGrad.resize(nOutputs, nInputs);
    std::uniform_real_distribution<float> dist(low,high);
    W = W.unaryExpr([&](float){return dist(gen);});
}

Eigen::MatrixXf Linear::forward(const Eigen::MatrixXf& input){
    if (input.size()!=B.size()){
        throw std::invalid_argument("Shape mismatch");
    }
    this->inputCache = input;
    Eigen::MatrixXf Z = W * input + B;
    return Z;
}
Eigen::MatrixXf Linear::backward(const Eigen::MatrixXf& gradOutputs){
    
}

std::vector<Eigen::MatrixXf*> Linear::getParameters(){
    return {&W,&B};
}

std::vector<Eigen::MatrixXf*> Linear::getGradients(){
    return {&wGrad, &bGrad};
}

