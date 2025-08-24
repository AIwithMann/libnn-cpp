#pragma once
#include<iostream>
#include<Eigen/Dense>
#include<random>
#include "Dataset.h"
#include "Model.h"
#include<unordered_map>
#include "Types.h"
#include<memory>

class Model;

struct ParamState{
    Eigen::MatrixXf v;
    Eigen::MatrixXf m;
    Eigen::MatrixXf u;
    Eigen::MatrixXf G;
    int t;
};
typedef std::unordered_map<Eigen::MatrixXf*, ParamState> State;
std::random_device rd;
std::mt19937 gen(rd());

enum class Optimizers{
    SGD,
    ADAM,
    RMSPROP,
    MOMENTUM,
    ADAGRAD
};

class Optimizer{
friend class Model;
protected:
    State state;
public:
    virtual void updateParams() = 0;
    virtual void init_state() = 0;
};

class SGD: public Optimizer{
    float lr;
    Trainables trainables;
public:
    SGD();
    void initialize(float lr, Trainables& trainables);
    void updateParams() override;
};

class Momentum: public Optimizer {
    float lr;
    float beta;
    Trainables& trainables;
public:
    Momentum();
    void initialize(float lr, float beta, Trainables& trainables);
    void init_state() override;
    void updateParams() override;
};

class ADAGRAD: public Optimizer{
    float lr;
    Trainables& trainables;
public:
    ADAGRAD();
    void initialize(float lr, Trainables& trainables);
    void init_state() override;
    void updateParams() override;
};

class RMSPROP: public Optimizer{
    float lr;
    float dRate;
    Trainables& trainables;
public:
    RMSPROP();
    void initialize(float lr, float dRate, Trainables& trainables);
    void init_state() override;
    void updateParams() override;
};

class ADAM: public Optimizer{
    float lr; 
    float Beta1;
    float Beta2;
    Trainables& trainables;
public:
    ADAM();
    void initialize(float lr, float Beta1, float Beta2, Trainables& trainables);
    void init_state() override;
    void updateParams() override;
};
