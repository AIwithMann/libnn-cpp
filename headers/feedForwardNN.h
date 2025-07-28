#pragma once
#include "Layer.h"

class FeedForwardNN {
public:
    FeedForwardNN(int nInputs, int nOutputs, float dropout);
    
    void addLayer(int nOutputs, ActivationFunction activation);
    void forward(Eigen::VectorXf inputs);
    const Eigen::VectorXf& getOutput() const;


private:
    int numInputs;
    int numOutputs;
    int numLayers;
    int LastLayerIdx;
    float dropout;
    bool istraining;

    std::vector<Layer> Network;
    Eigen::VectorXf Output;
};
