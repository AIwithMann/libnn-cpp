#include "Model.h"
#include "Dataset.h"
#include "Loss.h"

Model::Model(int nInputs, int nOutputs,Dataset& ds, Optimizers optimType, Loss lt): numInputs(nInputs), numOutputs(nOutputs), dataset(ds), lossType(lt){
    switch(optimType){
        case Optimizers::SGD:
            optim = std::make_shared<SGD>();
            break;
        case Optimizers::RMSPROP:
            optim = std::make_shared<RMSPROP>();
            break;
        case Optimizers::MOMENTUM:
            optim = std::make_shared<Momentum>();
            break;
        case Optimizers::ADAM:
            optim = std::make_shared<ADAM>();
            break;
        case Optimizers::ADAGRAD:
            optim = std::make_shared<ADAGRAD>();
            break;
    }
}

void Model::setTraining(bool mode){ this->isTraining = mode; }

Eigen::MatrixXf& Model::forward(Eigen::MatrixXf& X) {
    if (X.rows()!= 1 || X.cols() != numInputs) throw std::runtime_error("Shape mismatch");
    Eigen::MatrixXf* current = &X;
    for(auto& layer: Layers){
        current = &layer->forward(*current);
    }
    Output = *current;
    return Output;
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
        optim->init_state();
        optim->updateParams();
        for (auto& [param, state]: optim->state){
            state.t++;
        }
    }
    else if(optim->state[0].t > 0){
        optim->updateParams();
        optim->state[0].t++;
        for (auto& [param,state]: optim->state){
            state.t++;
        }
    }
    else{
        throw std::runtime_error("Optimizer state not initialized");
    }
}

void Model::addLayer(std::shared_ptr<Layer> l){
    Layers.push_back(l);
    numLayers++;
}

void Model::train(int epochs, bool showStats ){
    if (!isTraining) throw std::runtime_error("Model not in training mode");

    for(int i = 0; i < epochs; i++){
        dataset.shuffleData();
        for(int i = 0; i < (dataset.getNumSamples()/dataset.getBatchSize()); i++){
            Batch batch = dataset.getBatch(i);
            Eigen::MatrixXf out = forward(batch.batchX);
            float loss = calculateLoss(out, batch.batchY);
            backward(dataset.getBatchSize());
            updateParams();
            if(showStats) std::cout << "loss: " << loss << "\n";
        }
    }
}
