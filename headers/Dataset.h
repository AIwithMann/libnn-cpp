#pragma once

#include<iostream>
#include<Eigen/Dense>
#include<cmath>
#include<vector>
#include<numeric>
#include<random>
#include<algorithm>
#include<optional>

bool shapeMismatch(Eigen::MatrixXf x, Eigen::MatrixXf y);

class Dataset{
private:
    int numSamples;
    int numClasses;
    int batchSize;
    Eigen::MatrixXf X;
    Eigen::MatrixXf Y;
    bool shuffle;

    struct Batch{
        Eigen::MatrixXf batchX;
        Eigen::MatrixXf batchY;
    };

public:
    Dataset(Eigen::MatrixXf Samples, Eigen::MatrixXf Labels, bool Shuffle, int batchS);
    inline int getBatchSize() const;
    inline Batch getBatch(int batchidx) const;
    inline void shuffleData(std::optional<unsigned int> seed = std::nullopt);
};