#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>

enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH
};

class Layer {
public:
    Layer(int numInputs, int numOutputs, ActivationFunction activationFn, bool initialize = true);
    
    void forward(const Eigen::VectorXf& inputVec, bool istraining);
    void modifyDropout(float p = 1.0f);
    const Eigen::VectorXf& getOutput() const;
    int getnumOutputs() const;

private:
    int numInputs;
    int numOutputs;
    float dropoutRate;

    Eigen::MatrixXf WEIGHTS;
    Eigen::VectorXf BIASES;
    Eigen::VectorXf INPUT;
    Eigen::VectorXf OUTPUT;

    ActivationFunction ACTIVATION;
    void applyActivation();
    void applyDropout();

    static float relu(float x);
    static float sigmoid(float x);
    static float tanh_fn(float x);
};
