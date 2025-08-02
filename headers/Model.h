#include<iostream>
#include<vector>
#include<Eigen/Dense>
#include<optional>
#include "Layers.h"
#include "Loss.h"
#include "Dataset.h"
#include<memory>
#include<string>
class Model{
private:
    int numLayers;
    int numInputs;
    int numOutputs;
    bool isTraining;
    std::vector<std::unique_ptr<Layer>> Layers;
    Eigen::MatrixXf Output;
    float loss;
    Dataset& dataset;
    Loss lossType;
    void backward();
    void updateParams();
    std::vector<Eigen::MatrixXf&> Gradients();

    
public:
    Model(int nInputs, int nOutputs, Dataset& ds);
    void setTraining(bool mode);
    void train(int epochs);
    Eigen::MatrixXf& forward(Eigen::MatrixXf& X);
    float calculateLoss(Eigen::MatrixXf& Ypred, Eigen::MatrixXf& Y);
    void backward();
};