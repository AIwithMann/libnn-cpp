#pragma once

#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include<vector>
#include<numeric>
#include<random>
#include<algorithm>
#include<optional>

bool shapeMismatch(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y);

class Dataset{
private:
    int numSamples;
    int numClasses;
    int batchSize;
    Eigen::MatrixXf& X;
    Eigen::MatrixXf& Y;
    bool shuffle;

    struct Batch{
        Eigen::MatrixXf batchX;
        Eigen::MatrixXf batchY;
    };
    std::vector<Batch&> Batches;

public:
    Dataset(const Eigen::MatrixXf Samples, const Eigen::MatrixXf Labels, bool Shuffle, int batchS);
    int getBatchSize() const;
    Batch getBatch(int batchidx) const;
    void shuffleData(std::optional<unsigned int> seed = std::nullopt);
    int getNumSamples() const;
};
