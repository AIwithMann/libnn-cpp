#include <iostream>
#include <Eigen/Dense>

enum class Loss{
    MSE,
    MAE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
};

bool shapeMismatch(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted){
    return True.size() != Predicted.size();
}
bool shapeMismatch(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted){
    return True.rows() != Predicted.rows() || True.cols() != Predicted.cols();
}

inline float MSE(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    return (True - Predicted).array().square().mean();
}

inline float MAE(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted){
    if (shapeMismatch(True, Predicted)){
        throw std::logic_error("shape mismatch");
    }
    return (True - Predicted).array().abs().mean();
}

inline float BinaryCE(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted){
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
    Eigen::ArrayXXf Y_hat = Predicted.array().min(1.0f - eps).max(eps);
    Eigen::ArrayXXf Y = True.array();
    Eigen::ArrayXXf log_probs = Y * Y_hat.log();

    return -log_probs.rowwise().sum().mean(); 
}