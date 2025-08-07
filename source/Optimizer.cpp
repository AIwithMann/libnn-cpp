#include "Optimizer.h"
#include "Model.h"


SGD::SGD(float lr, Trainables& trainables): lr(lr), trainables(trainables){}

void SGD::update(){
    for(auto& [param, grad]: trainables){
        *param -= lr * *grad;
        grad->setZero();
    }
}

Momentum::Momentum(float lr, float beta, Trainables& trainables): lr(lr), beta(beta), trainables(trainables){}

void Momentum::init_state(){
    for(auto& [param, grad]: trainables){
        ParamState ps;
        ps.v = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = std::move(ps);
    }
}
void Momentum::init_state(){
    for (auto& [param,grad]: trainables){
        ParamState ps;
        ps.v = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void Momentum::update(){
    for (auto& [param, grad] : trainables) {
        auto& ps = state[param];
        ps.v = beta * ps.v + *grad;
        *param -= lr * ps.v;
        grad->setZero();
    }
}


NAG::NAG(float lr, float beta, Trainables& trainables, Model& model): lr(lr), beta(beta), trainables(trainables), model(model){}

void NAG::init_state(){
    for (auto& [param,grad]:trainables){
        ParamState ps;
        ps.v = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void NAG::update(Batch& batch){
    std::vector<Eigen::MatrixXf> ACTUAL;
    for(auto& [param, grad] : trainables){
        ACTUAL.push_back(*param);
        auto& ps = state[param];
        *param -= beta* ps.v;
    
    }
    model.forward(batch.batchX);
    model.backward(batch.batchSize);
    int i = 0;
    for(auto& [param, grad] : trainables){
        auto& ps = state[param];
        *param = ACTUAL[i];
        ps.v = beta * ps.v + *grad;
        *param -= lr * ps.v;
        grad->setZero();
        i++;
    }
}