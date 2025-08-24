#pragma once

#include <Eigen/Dense>
#include <vector>

class Layer {
protected:
    bool isTraining = true;
    Eigen::MatrixXf inputCache;
    Eigen::MatrixXf output;

public:
    virtual ~Layer() = default;

    virtual void setTraining(bool mode);
    virtual bool isTrainable() const;
    virtual Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) = 0;
    virtual Eigen::MatrixXf backward(Eigen::MatrixXf& gradOutput) = 0;
    virtual std::vector<Eigen::MatrixXf*> getParameters();
    virtual std::vector<Eigen::MatrixXf*> getGradients();
};

class Linear : public Layer {
private:
    Eigen::MatrixXf W, B;       
    Eigen::MatrixXf wGrad, bGrad;

public:
    Linear(int nInputs, int nOutputs);
    bool isTrainable()const{return true;};
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(Eigen::MatrixXf& gradOutputs) override;
    std::vector<Eigen::MatrixXf*> getParameters();
    std::vector<Eigen::MatrixXf*> getGradients();
    int getNumOutputs() const { return W.rows(); }
    int getNumInputs() const { return W.cols(); }
};

class ReLU : public Layer {
public:
    bool isTrainable() const{ return false; }
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(Eigen::MatrixXf& gradOutput) override;
};
class Sigmoid : public Layer {
public:
    bool isTrainable() const{ return false; }
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(Eigen::MatrixXf& gradOutput) override;
};

class Tanh : public Layer {
public:
    bool isTrainable() const{ return false; }
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(Eigen::MatrixXf& gradOutput) override;
};

class Dropout: public Layer{
private:
    Eigen::MatrixXf mask;
    float p;
public: 
    bool isTrainable() const{ return false;}
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(Eigen::MatrixXf& gradOutput) override;
};
