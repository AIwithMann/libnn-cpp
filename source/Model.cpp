#include "Model.h"
#include "Dataset.h"

Model::Model(int nInputs, int nOutputs,Dataset& ds): numInputs(nInputs), numOutputs(nOutputs), dataset(ds){}

void Model::setTraining(bool mode){ this->isTraining = mode; }

Eigen::MatrixXf& Model::forward(Eigen::MatrixXf& X){
    if (X.rows()!= 1 || X.cols() != this->numInputs){
        throw std::invalid_argument("Shape mismatch\n");
    }
    Eigen::MatrixXf Y;
    for(auto& layer:Layers){
        Y = layer->forward(X);
    } 
    Output = Y;
    return Y;
}

float Model::calculateLoss(Eigen::MatrixXf& Ypred, Eigen::MatrixXf& Y){
    switch (lossType){
    case Loss::MSE:
        return MSE(Y, Ypred);
    
    case Loss::MAE:
        return MAE(Y,Ypred);

    case Loss::BINARY_CROSS_ENTROPY:
        return BinaryCE(Y,Ypred);

    case Loss::CATEGORICAL_CROSS_ENTROPY:
        return MultiCE(Y,Ypred);
    
    default:
        throw std::invalid_argument("Loss type must be from MSE, MAE, BINARY_CROSS_ENTROPY or CATEGORICAL_CROSS_ENTROPY");
    }
}

