#include <iostream>
#include <string>
#include <cmath>
#include <random>
#include <vector>
#include "/usr/include/eigen3/Eigen/Dense"

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

    Eigen::MatrixXf weights;   // [numOutputs x numInputs]
    Eigen::VectorXf biases;    // [numOutputs x 1]
    
    Eigen::VectorXf input;     // [numInputs x 1]
    Eigen::VectorXf output;    // [numOutputs x 1]

    ActivationFunction activation;

    // === Activation functions ===
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
                output[i] /= (1.0f - dropoutRate);  // Inverted dropout scaling
            }
        }
    }
 
public:
    Layer(int numInputs, int numOutputs, ActivationFunction activationFn, bool initialize = true)
        : numInputs(numInputs), numOutputs(numOutputs), activation(activationFn){
        weights = Eigen::MatrixXf(numOutputs, numInputs);
        biases = Eigen::VectorXf::Zero(numOutputs);
        input = Eigen::VectorXf(numInputs);
        output = Eigen::VectorXf(numOutputs);

        if (initialize) {
            // He or Xavier init — here simple scaled uniform
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int i = 0; i < weights.rows(); ++i) {
                for (int j = 0; j < weights.cols(); ++j) {
                    weights(i, j) = dist(gen) * std::sqrt(2.0f / numInputs);  // He init for ReLU
                }
            }
        }
    }
    
    const int getnumOutputs() const{
        return this->numOutputs;
    }
    void modifyDropout(float p = 1.0){
        this->dropoutRate = p;
    }
    void forward(const Eigen::VectorXf& inputVec, bool istraining){
        
            if (inputVec.size() != numInputs) {
                std::cerr << "Input size mismatch: expected " << numInputs << ", got " << inputVec.size() << "\n";
                return;
            }

            input = inputVec;
            output = (weights * input) + biases;
            applyActivation();
            if (istraining){
                applyDropout();
            }
            return;
        
    }
    const Eigen::VectorXf& getOutput() const {
        return output;
    }
    
};

class feedForwardNN{
private:
    int numInputs;
    int numOutputs;
    int numLayers = 0;
    int LastLayerIdx = -1;

    float dropout;
    bool istraining;

    std::vector<Layer> Network;
    Eigen::VectorXf Output;
    
public:
    feedForwardNN(int nInputs, int nOutputs, float dropout): numInputs(nInputs), numOutputs(nOutputs), dropout(dropout){
        Output.resize(numOutputs);
    }
    
    void train(){this-> istraining = true;}
    void valid(){this-> istraining = false;}

    void addLayer(int nOutputs,ActivationFunction activation){
        int nInputs = Network[LastLayerIdx].getnumOutputs();
        Layer newLayer = Layer(nInputs, nOutputs, activation);
        newLayer.modifyDropout(dropout);
        Network.push_back(std::move(newLayer));
        numLayers++;
        LastLayerIdx++;
    }

    void forward(Eigen::VectorXf inputs){
        if (inputs.size() != numInputs){
            std::cerr << "dimension mismatch\n";
            return;
        }

        for(int i = 1; i < numLayers; i++){
            Network[i].forward(inputs, istraining);
            inputs = Network[i].getOutput();
        }
        Output = inputs;
    }

    const Eigen::VectorXf& getOutput() const{
        return Output;
    }

    void loss()
};
