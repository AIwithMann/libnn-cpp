#include "Layers.h"
#include <random>
#include <stdexcept>
#include <cmath>

static std::random_device rd;
static std::mt19937 gen(rd());

void Layer::setTraining(bool mode) {
    isTraining = mode;
}

Linear::Linear(int nInputs, int nOutputs, float dropout)
    : dropout(dropout) {
    float limit = std::sqrt(6.0f / (nInputs + nOutputs));
    std::uniform_real_distribution<float> dist(-limit, limit);

    W = Eigen::MatrixXf::NullaryExpr(nOutputs, nInputs, [&]() { return dist(gen); });
    B = Eigen::MatrixXf::Zero(nOutputs, 1);

    wGrad = Eigen::MatrixXf::Zero(nOutputs, nInputs);
    bGrad = Eigen::MatrixXf::Zero(nOutputs, 1);
}

void Linear::applyDropout(Eigen::MatrixXf& Z) {
    if (dropout <= 0.0f) return;
    std::bernoulli_distribution dist(1.0f - dropout);
    auto bernoulli = [&]() { return dist(gen); };
    dropoutMask = Eigen::MatrixXf::NullaryExpr(Z.rows(), Z.cols(), bernoulli).cast<float>();
    Z = Z.cwiseProduct(dropoutMask);
}

Eigen::MatrixXf& Linear::forward(const Eigen::MatrixXf& input) {
    if (input.rows() != W.cols()) {
        throw std::invalid_argument(
            "Linear::forward - Expected input rows = " + std::to_string(W.cols()) +
            ", got " + std::to_string(input.rows())
        );
    }

    inputCache = input;
    Eigen::MatrixXf Z = (W * input).colwise() + B.col(0);
    output = Z;
    return output;
}

Eigen::MatrixXf Linear::backward(const Eigen::MatrixXf& gradOutput){
    wGrad = gradOutput * inputCache.transpose();
    bGrad = gradOutput.rowwise().sum();
    return W.transpose() * gradOutput;
}
std::vector<Eigen::MatrixXf*> Linear::getParameters() {
    return { &W, &B };
}

std::vector<Eigen::MatrixXf*> Linear::getGradients() {
    return { &wGrad, &bGrad };
}

Eigen::MatrixXf& ReLU::forward(const Eigen::MatrixXf& input) {
    inputCache = input;
    this->output = input.array().max(0.0f);
    return this->output;
}

Eigen::MatrixXf ReLU::backward(const Eigen::MatrixXf& gradOutput) {
    return gradOutput.array() * (output.array() > 0.0f).cast<float>();
}

Eigen::MatrixXf& Sigmoid::forward(const Eigen::MatrixXf& input) {
    inputCache = input;
    output = input.unaryExpr([](float x) {
        if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
        float expX = std::exp(x);
        return expX / (1.0f + expX);
    });
    return output;
}

Eigen::MatrixXf Sigmoid::backward(const Eigen::MatrixXf& gradOutput){
    auto sig = output.array();
    return gradOutput.array() * (sig * (1 - sig));
}

Eigen::MatrixXf& Tanh::forward(const Eigen::MatrixXf& input) {
    inputCache = input;
    output = input.array().tanh();
    return output;
}

Eigen::MatrixXf Tanh::backward(const Eigen::MatrixXf& gradOutputs){
    return gradOutputs.array() * (1 - output.array().square());
}
