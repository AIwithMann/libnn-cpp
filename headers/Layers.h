#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH
};

class Layer {
protected:
    bool isTraining = true;
    Eigen::MatrixXf inputCache; //Batchsize * nInputs
    Eigen::MatrixXf output; //Batchsize * nOutputs
    
public:
    virtual ~Layer() = default;
    virtual void setTraining(bool mode);
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutput);
    virtual void update(float learning_rate);
    virtual std::vector<Eigen::MatrixXf*> getParameters();
    virtual std::vector<Eigen::MatrixXf*> getGradients();
};

class Linear:public Layer{
private:
    Eigen::MatrixXf W, B, wGrad, bGrad,dropoutMask;
    float dropout;
    inline void applyDropout(Eigen::MatrixXf& Z);
public:
    inline Linear(int nInputs, int nOutputs, float low, float high, float dropout);
    inline ~Linear()=default;
    inline Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    inline Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutputs);
    inline std::vector<Eigen::MatrixXf*> getParameters();
    inline std::vector<Eigen::MatrixXf*> getGradients();
};
