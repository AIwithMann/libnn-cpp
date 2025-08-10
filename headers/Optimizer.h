#pragma once
#include<iostream>
#include<Eigen/Dense>
#include<random>
#include "Dataset.h"

typedef std::vector<std::pair<Eigen::MatrixXf*, Eigen::MatrixXf*>> Trainables;
typedef std::unordered_map<Eigen::MatrixXf*, ParamState> State;
std::random_device rd;
std::mt19937 gen(rd());

enum class Optimizers{
    SGD,
    ADAM,
    RMSPROP,
    MOMENTUM,
    NAG,
    ADAGRAD
};

struct ParamState{
    Eigen::MatrixXf v;
    Eigen::MatrixXf m;
    Eigen::MatrixXf u;
    Eigen::MatrixXf G;
    int t;
};

class Optimizer{
friend class Model;
protected:
    State state;
public:
    virtual void update() = 0;
    virtual void init_state(Model& model) = 0;
};

class SGD: public Optimizer{
    float lr;
    Trainables trainables;
public:
    SGD(float lr, Trainables& traiables);
    void update();
};

class Momentum: public Optimizer {
    float lr;
    float beta;
    Trainables& trainables;
public:
    Momentum(float lr, float beta, Trainables& trainables);
    void init_state();
    void update();
};

class NAG: public Optimizer {
    float lr;
    float beta;
    Trainables& trainables;
    Model& model;
public:
    NAG(float lr, float beta, Trainables& trainables, Model& model);
    void init_state();
    void update(Batch& batch);
};

class ADAGRAD: public Optimizer{
    float lr;
    Trainables& trainables;
public:
    ADAGRAD(float lr, Trainables& trainables);
    void init_state();
    void update();
};

class RMSPROP: public Optimizer{
    float lr;
    float dRate;
    Trainables& trainables;
public:
    RMSPROP(float lr, float dRate, Trainables& Trainables);
    void init_state();
    void update();
};

class ADAM: public Optimizer{
    float lr; 
    float Beta1;
    float Beta2;
    Trainables& trainables;
public:
    ADAM(float lr, float Beta1, float Beta2, Trainables& trainables);
    void init_state();
    void update();
};
