#pragma once
#include<iostream>
#include<vector>
#include<Eigen/Dense>
#include<optional>
#include "Layers.h"
#include "Loss.h"
#include "Dataset.h"
#include "Optimizer.h"
#include<memory>
#include<string>
#include<any>
#include "Types.h"

class Optimizer;
enum class Optimizers;

class Model{
private:
    int numLayers;
    int numInputs;
    int numOutputs;
    bool isTraining;
    std::vector<std::shared_ptr<Layer>> Layers;
    Eigen::MatrixXf Output;
    float loss;
    Dataset& dataset;
    Loss lossType;
    Trainables trainables;
    std::shared_ptr<Optimizer> optim;  
public:
    Model(int nInputs, int nOutputs, Dataset& ds, Optimizers optimType, Loss lt);
    void setTraining(bool mode);
    Eigen::MatrixXf& forward(Eigen::MatrixXf& X);
    float calculateLoss(Eigen::MatrixXf& Ypred, Eigen::MatrixXf& Y);
    void backward(size_t batchIdx);
    Eigen::MatrixXf lossGradient(Eigen::MatrixXf& Ypred, Eigen::MatrixXf& Y);
    std::vector<std::shared_ptr<Layer>>& getLayers();
    int getNumLayers();
    int getNumInputs();
    int getNumOutputs();
    void updateParams();
    Trainables& getTrainables();
    void addLayer(std::shared_ptr<Layer> l);
    void train(int epochs, bool showStates =true);
};
