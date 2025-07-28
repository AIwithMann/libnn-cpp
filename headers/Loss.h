#pragma once
#include <iostream>
#include <Eigen/Dense>

enum class Loss{
    MSE,
    MAE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
};

bool shapeMismatch(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted);
bool shapeMismatch(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);
inline float MSE(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted);
inline float MAE(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted);
inline float BinaryCE(const Eigen::VectorXf& True, const Eigen::VectorXf& Predicted);
inline float MultiCE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);