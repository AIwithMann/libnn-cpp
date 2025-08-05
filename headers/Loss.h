#pragma once

#include <iostream>
#include <Eigen/Dense>

enum class Loss{
    MSE,
    MAE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
};

bool shapeMismatch(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);
inline float MSE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);
inline float MAE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);
inline float BinaryCE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);
inline float MultiCE(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted);
inline Eigen::MatrixXf MSEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize);
inline Eigen::MatrixXf MAEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize);
inline Eigen::MatrixXf BinaryCEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize);
inline Eigen::MatrixXf MultiCEgrad(const Eigen::MatrixXf& True, const Eigen::MatrixXf& Predicted, int bSize);
