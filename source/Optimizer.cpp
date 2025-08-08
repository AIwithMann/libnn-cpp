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

ADAGRAD::ADAGRAD(float lr, Trainables& trainables): lr(lr), trainables(trainables){}

void ADAGRAD::init_state(){
    for (auto& [param, grad]: trainables){
        ParamState ps;
        ps.G = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void ADAGRAD::update(){
    for(auto& [param,grad] : trainables){
        auto& ps = state[param];
        ps.G += *grad * *grad;
        *param -= lr * *grad / (ps.G.array().sqrt() + 1e-7);
        grad->setZero();
    }
}   

RMSPROP::RMSPROP(float lr, float dRate, Trainables& trainables): lr(lr), dRate(dRate), trainables(trainables){}

void RMSPROP::init_state(){
    for(auto& [param,grad]: trainables){
        ParamState ps;
        ps.G = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void RMSPROP::update(){
    for(auto& [param, grad]: trainables){
        auto& ps = state[param];
        ps.G = dRate * ps.G.array() + (1 - dRate) * grad->array().square();
        auto denom = (ps.G.array() + 1e-7).sqrt();
        *param -= lr * grad->array() / denom;
        grad->setZero();
    }
}

ADAM::ADAM(float lr, float Beta1, float Beta2, Trainables& trainables):lr(lr), Beta1(Beta1), Beta2(Beta2), trainables(trainables){}

void ADAM::init_state(){
    for (auto& [param, grad]: trainables){
        ParamState ps;
        ps.m = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        ps.u = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void ADAM::update(){
    for(auto& [param, grad]: trainables){
        auto& ps = state[param];
        ps.m = Beta1 * ps.m + (1 - Beta1) * grad->array();
        ps.u = Beta2 * ps.u + (1 - Beta2) * grad->array().square();
        ps.t++;
        ps.m /= 1 - std::pow(Beta1, ps.t);
        ps.u /= 1 - std::pow(Beta2, ps.t);
        *param -= lr * ps.m / (ps.u.array().sqrt() + 1e-7);
        grad->setZero();
    }
}
