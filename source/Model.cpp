#include "Model.h"
#include "Dataset.h"
#include "Loss.h"

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
    switch(lossType){
    case Loss::MSE:
        return MSE(Y, Ypred);
    
    case Loss::MAE:
        return MAE(Y,Ypred);

    case Loss::BINARY_CROSS_ENTROPY:
        return BinaryCE(Y,Ypred);

    case Loss::CATEGORICAL_CROSS_ENTROPY:
        return MultiCE(Y,Ypred);
    
    default:
        throw std::invalid_argument("Loss    type must be from MSE, MAE, BINARY_CROSS_ENTROPY or CATEGORICAL_CROSS_ENTROPY");
    }
}

std::vector<std::shared_ptr<Layer>>& Model::getLayers(){
    return this->Layers;
}

void Model::backward(size_t batchIdx){
    Eigen::MatrixXf grad;
    switch(lossType){
        case Loss::MSE:
            grad =  MSEgrad(Output, dataset.getBatch(batchIdx).batchY, dataset.getBatchSize());
            break;
        case Loss::MAE:
            grad = MAEgrad(Output, dataset.getBatch(batchIdx).batchY, dataset.getBatchSize());
            break;
        case Loss::BINARY_CROSS_ENTROPY:
            grad = BinaryCEgrad(Output, dataset.getBatch(batchIdx).batchY, dataset.getBatchSize());
            break;
        case Loss::CATEGORICAL_CROSS_ENTROPY:
            grad = MultiCEgrad(Output, dataset.getBatch(batchIdx).batchY, dataset.getBatchSize());
            break;
        default:
            throw std::invalid_argument("Loss type must be from MSE, MAE, BINARY_CROSS_ENTROPY or CATEGORICAL_CROSS_ENTROPY");
    }
    for(int i = numLayers-1; i>=0; --i){
        grad = Layers[i]->backward(grad);
    }
}

Trainables& Model::getTrainables(){
    for(auto& layer:Layers){
        if (!layer->isTrainable()) continue;
        auto params = layer->getParameters();
        auto grads = layer->getGradients();
        for(int i = 0; i < params.size(); ++i){
            trainables.push_back({params[i], grads[i]});
        }
    }
    return trainables;
}
Eigen::MatrixXf Model::lossGradient(Eigen::MatrixXf& Ypred, Eigen::MatrixXf& Y){
    switch(lossType){
        case Loss::MSE:
            return MSEgrad(Ypred, Y, 1);
            break;
        case Loss::MAE:
            return MAEgrad(Ypred, Y, 1);
            break;
        case Loss::BINARY_CROSS_ENTROPY:
            return BinaryCEgrad(Ypred, Y, 1);
            break;
        case Loss::CATEGORICAL_CROSS_ENTROPY:
            return MultiCEgrad(Ypred, Y, 1);
            break;
        default:
            throw std::invalid_argument("Loss type must be from MSE, MAE, BINARY_CROSS_ENTROPY or CATEGORICAL_CROSS_ENTROPY");
    }
}

int Model::getNumLayers(){return numLayers;}
int Model::getNumInputs(){return numInputs;}
int Model::getNumOutputs(){return numOutputs;}

void Model::updateParams(){
    if (optim->state[0].t == 0){
        optim->init_state(*this);
        optim->state[0].t++;
        optim->update();
    }
    else if(optim->state[0].t > 0){
        optim->update();
    }
    else{
        throw std::runtime_error("Optimizer state not initialized");
    }
}

