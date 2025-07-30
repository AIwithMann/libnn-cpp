#pragma once

#include <Eigen/Dense>

enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH
};

class Linear {
public:
    Linear(int numInputs, int numOutputs, ActivationFunction activationFn, bool initialize = true);

    void forward(const Eigen::VectorXf& inputVec, bool isTraining);
    const Eigen::VectorXf& getOutput() const;
    int getnumOutputs() const;

private:
    int numInputs;
    int numOutputs;
    float dropoutRate;

    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf input;
    Eigen::VectorXf output;

    ActivationFunction activation;

    void applyActivation();
    void applyDropout();
};
