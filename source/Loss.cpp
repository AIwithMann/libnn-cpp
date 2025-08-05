#include <iostream>
#include <Eigen/Dense>

enum class Loss{
    MSE,
    MAE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
};

bool shapeMismatch(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted){
    return True.rows() != Predicted.rows() || True.cols() != Predicted.cols();
}

inline float MSE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    return (True - Predicted).array().square().mean();
}

inline float MAE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    return (True - Predicted).array().abs().mean();
}

inline float BinaryCE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    float eps = 1e-7;
    Eigen::ArrayXf Y = True.array().min(1-eps).max(eps);
    Eigen::ArrayXf Yhat = Predicted.array().min(1-eps).max(eps);
    
    Eigen::ArrayXf term1 = Y*Eigen::log(Yhat);
    Eigen::ArrayXf term2 = (1-Y)*Eigen::log(1-Yhat);

    return -(term1 + term2).mean();
}

inline float MultiCE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }

    const float eps = 1e-7f;
    Eigen::ArrayXf Y_hat = Predicted.array().min(1.0f - eps).max(eps);
    Eigen::ArrayXf Y = True.array();
    Eigen::ArrayXf log_probs = Y * Y_hat.log();

    return -log_probs.rowwise().sum().mean(); 
}

inline Eigen::MatrixXf MSEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    Eigen::ArrayXf grad = Predicted.array() - True.array();
    grad *= 2;
    grad /= bSize;
    return grad.matrix();
}

inline Eigen::MatrixXf MAEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    Eigen::ArrayXf grad = Predicted - True;
    grad = grad.sign();
    grad /= bSize;
    return grad.matrix();
}

inline Eigen::MatrixXf BinaryCEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    Eigen::ArrayXf grad = (Predicted.array() - True.array())/Predicted.array()* (1- Predicted.array());
    grad /= bSize;
    return grad.matrix();
}

inline Eigen::MatrixXf MultiCEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    Eigen::ArrayXXf grad = Predicted.array() - True.array();
    grad /= bSize;
    return grad.matrix();
}
