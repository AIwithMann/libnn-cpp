#include<iostream>
#include<vector>
#include<Eigen/Dense>
#include "Loss.h"
#include "Layer.h"

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

    void calculate_loss(Eigen::VectorXf actualY){
        //Accepts input in batch
        
        
    }
};
