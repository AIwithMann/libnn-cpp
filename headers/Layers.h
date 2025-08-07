#pragma once

#include <Eigen/Dense>
#include <vector>

class Layer {
protected:
    bool isTraining = true;
    Eigen::MatrixXf inputCache;
    Eigen::MatrixXf output;
    bool istrainable = true;

public:
    virtual ~Layer() = default;

    virtual void setTraining(bool mode);
    virtual Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) = 0;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutput) = 0;
    virtual void update(float learningRate) {}
    virtual std::vector<Eigen::MatrixXf*> getParameters() { return {}; }
    virtual bool isTrainable() const { return isTrainable; }
    virtual std::vector<Eigen::MatrixXf*> getGradients() { return {}; }
};

class Linear : public Layer {
private:
    Eigen::MatrixXf W, B;       
    Eigen::MatrixXf wGrad, bGrad;
    Eigen::MatrixXf dropoutMask;
    float dropout = 0.0f;
    bool istrainable = true;

    void applyDropout(Eigen::MatrixXf& Z);

public:
    Linear(int nInputs, int nOutputs, float dropout = 0.0f);
    bool isTrainable();
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutputs) override;
    void update(float learningRate) override;
    bool isTrainable() const override{ return istrainable; }
    std::vector<Eigen::MatrixXf*> getParameters() override;
    std::vector<Eigen::MatrixXf*> getGradients() override;
    int getNumOutputs() const { return W.rows(); }
    int getNumInputs() const { return W.cols(); }
};

class ReLU : public Layer {
private:
    bool istrainable = false;
public:
    bool isTrainable() const override{ return istrainable; }
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutput) override;
};
class Sigmoid : public Layer {
private:
    bool istrainable = false;
public:
    bool isTrainable() const override{ return istrainable; }
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutput) override;
};

class Tanh : public Layer {
private:
    bool istrainable = false;
public:
    bool isTrainable() const override{ return istrainable; }
    Eigen::MatrixXf& forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& gradOutput) override;
};

