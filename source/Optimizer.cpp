#include "/home/aiwithmann/Desktop/Forever-Beta/libnn-cpp/headers/Optimizer.h"
#include "/home/aiwithmann/Desktop/Forever-Beta/libnn-cpp/headers/Model.h"


SGD::SGD(){}

void SGD::initialize(float lr, Trainables& trainables){
    this->lr = lr;
    this->trainables = trainables;
}

void SGD::updateParams(){
    for(auto& [param, grad]: trainables){
        *param -= lr * *grad;
        grad->setZero();
    }
}

Momentum::Momentum(){}

void Momentum::initialize(float lr, float beta, Trainables& trainables){
    this->lr = lr;
    this->beta = beta;
    this->trainables = trainables;
}

void Momentum::init_state(){
    for(auto& [param, grad]: trainables){
        ParamState ps;
        ps.v = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = std::move(ps);
    }
}

void Momentum::updateParams(){
    for (auto& [param, grad] : trainables) {
        auto& ps = state[param];
        ps.v = beta * ps.v + *grad;
        *param -= lr * ps.v;
        grad->setZero();
    }
}

ADAGRAD::ADAGRAD(){}

void ADAGRAD::initialize(float lr, Trainables& trainables){
    this->lr = lr;
    this->trainables = trainables;
}

void ADAGRAD::init_state(){
    for (auto& [param, grad]: trainables){
        ParamState ps;
        ps.G = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void ADAGRAD::updateParams(){
    for(auto& [param,grad] : trainables){
        auto& ps = state[param];
        ps.G.array() +=grad->array().square();
        param->array() -= lr * grad->array() / (ps.G.array().sqrt()+1e-7f);
        grad->setZero();
    }
}   

RMSPROP::RMSPROP(){}

void RMSPROP::initialize(float lr, float dRate, Trainables& trainables){
    this->lr = lr;
    this->dRate = dRate;
    this->trainables = trainables;
}

void RMSPROP::init_state(){
    for(auto& [param,grad]: trainables){
        ParamState ps;
        ps.G = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void RMSPROP::updateParams(){
    for(auto& [param, grad]: trainables){
        auto& ps = state[param];
        ps.G = dRate * ps.G.array() + (1 - dRate) * grad->array().square();
        auto denom = (ps.G.array() + 1e-7).sqrt();
        param->array() -= lr * grad->array() / denom;
        grad->setZero();
    }
}

ADAM::ADAM(){}

void ADAM::initialize(float lr, float Beta1, float Beta2, Trainables& trainables){
    this->lr = lr;
    this->Beta1 = Beta1;
    this->Beta2 = Beta2;
    this->trainables = trainables;
}

void ADAM::init_state(){
    for (auto& [param, grad]: trainables){
        ParamState ps;
        ps.m = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        ps.u = Eigen::MatrixXf::Zero(param->rows(), param->cols());
        state[param] = ps;
    }
}

void ADAM::updateParams(){
    for(auto& [param, grad]: trainables){
        auto& ps = state[param];
        ps.m = Beta1 * ps.m.array() + (1 - Beta1) * grad->array();
        ps.u = Beta2 * ps.u.array() + (1 - Beta2) * grad->array().square();
        ps.t++;
        ps.m /= 1 - std::pow(Beta1, ps.t);
        ps.u /= 1 - std::pow(Beta2, ps.t);
        param->array() -= lr * ps.m.array() / (ps.u.array().sqrt() + 1e-7);
        grad->setZero();
    }
}
