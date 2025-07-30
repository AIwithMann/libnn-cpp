#include "Layer.h"
#include <cmath>
#include <random>
#include<Eigen/Dense>
#include <stdexcept>

static std::random_device rd;
static std::mt19937 gen(rd());

Layer::Layer(int numInputs, int numOutputs, ActivationFunction activationFn, bool initialize)
    : numInputs(numInputs), numOutputs(numOutputs), activation(activationFn), dropoutRate(0.1f) {

    weights = Eigen::MatrixXf::Zero(numOutputs, numInputs);
    biases = Eigen::VectorXf::Zero(numOutputs);

    if (initialize) {
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(numInputs));
        for (int i = 0; i < weights.rows(); ++i)
            for (int j = 0; j < weights.cols(); ++j)
                weights(i, j) = dist(gen);
    }

    output = Eigen::VectorXf::Zero(numOutputs);
}

void Layer::forward(const Eigen::VectorXf& inputVec, bool isTraining) {
    if (inputVec.size() != numInputs)
        throw std::invalid_argument("Input vector size does not match layer input size.");

    input = inputVec;
    output = (weights * input + biases);
    applyActivation();

    if (isTraining) {
        applyDropout();
    }
}

void Layer::modifyDropout(float p) {
    if (p < 0.0f || p > 1.0f)
        throw std::invalid_argument("Dropout rate must be in [0, 1]");
    dropoutRate = p;
}

const Eigen::VectorXf& Layer::getOutput() const {
    return output;
}

int Layer::getnumOutputs() const {
    return numOutputs;
}

void Layer::applyActivation() {
    switch(this->activation){
        case ActivationFunction::RELU:
            output = output.array().max(0.0f);
            break;
        case ActivationFunction::SIGMOID:
            output = (1.0f / (1.0f + (-output.array()).exp())).matrix();
            break;
        case ActivationFunction::TANH:
            output = output.array().tanh();
            break;
        default:
            throw std::invalid_argument("Activation function is not from 'RELU', 'SIGMOID' or 'TANH'");

    }
}

void Layer::applyDropout() {
    if (dropoutRate <= 0.0f || dropoutRate >= 1.0f)
        return;

    std::bernoulli_distribution keepProb(1.0f - dropoutRate);
    for (int i = 0; i < output.size(); ++i) {
        if (!keepProb(gen)) {
            output[i] = 0.0f;
        } else {
            output[i] /= (1.0f - dropoutRate);  // Inverted dropout
        }
    }
}


