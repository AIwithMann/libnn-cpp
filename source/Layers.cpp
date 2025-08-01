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

Linear::Linear(int nInputs, int nOutputs, float low, float high, float dropout){
    this->B = Eigen::MatrixXf::Zero(nOutputs,1);
    this->bGrad.resize(nOutputs,1);
    this->W.resize(nOutputs,nInputs);
    this->wGrad.resize(nOutputs, nInputs);
    this->dropout = dropout;

    std::uniform_real_distribution<float> dist(low,high);
    W = W.unaryExpr([&](float){return dist(gen);});
}

Eigen::MatrixXf Linear::forward(const Eigen::MatrixXf& input){
    //Input must be nInputs * BatchSize
    if (input.rows()!=W.cols()){
        throw std::invalid_argument("Shape mismatch");
    }
    this->inputCache = input;
    Eigen::MatrixXf Z = (W * input).colwise() + B.col(0);
    if(isTraining) { applyDropout(Z); }
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

void Linear::applyDropout(Eigen::MatrixXf& Z){
    std::bernoulli_distribution dist(1.0f - this->dropout);
    auto bernouli = [&](){return dist(gen);};
    Eigen::MatrixXf DP = Eigen::MatrixXf::NullaryExpr(Z.rows(),Z.cols(), bernouli).cast<float>();
    this->dropoutMask = DP;
    Z = Z.cwiseProduct(DP);
}
