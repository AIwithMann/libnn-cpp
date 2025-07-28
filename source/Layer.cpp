#include <iostream>
#include <string>
#include <cmath>
#include <random>
#include <Eigen/Dense>

std::random_device rd;
std::mt19937 gen(rd());

enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH
};
enum class Loss{
    CROSSENTROPY,
    MEANSQUAREDERROR,
    MEANABSOLUTEERROR
};

class Layer{
private:
    int numInputs;
    int numOutputs;
    float dropoutRate = 0.1;

    Eigen::MatrixXf weights; 
    Eigen::VectorXf biases;
    
    Eigen::VectorXf input;
    Eigen::VectorXf output;

    ActivationFunction activation;


    static float relu(float x) {
        return x > 0 ? x : 0;
    }

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static float tanh_fn(float x) {
        return std::tanh(x);
    }

    void applyActivation() {
        for (int i = 0; i < output.size(); ++i) {
            switch (activation) {
                case ActivationFunction::RELU:
                    output[i] = relu(output[i]);
                    break;
                case ActivationFunction::SIGMOID:
                    output[i] = sigmoid(output[i]);
                    break;
                case ActivationFunction::TANH:
                    output[i] = tanh_fn(output[i]);
                    break;
                default:
                    std::cerr << "Unknown activation function.\n";
            }
        }
    }

    void applyDropout() {
        if (dropoutRate <= 0.0f || dropoutRate >= 1.0f) return;

        std::bernoulli_distribution keepProb(1.0f - dropoutRate);
        for (int i = 0; i < output.size(); ++i) {
            if (!keepProb(gen)) {
                output[i] = 0.0f;
            } else {
                output[i] /= (1.0f - dropoutRate);
            }
        }
    }
};