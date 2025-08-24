#include "Dataset.h"
#include <stdexcept>
#include <numeric>
#include <random>
#include <algorithm>

bool shapeMismatch(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y) {
    return X.rows() != Y.rows();
}

Dataset::Dataset(Eigen::MatrixXf Samples, Eigen::MatrixXf Labels,bool Shuffle, int batchS): X(Samples), Y(Labels),shuffle(Shuffle),batchSize(batchS),numSamples(Samples.rows()),numClasses(Labels.cols()){
    if (shapeMismatch(Samples, Labels)) {
        throw std::logic_error("Shape mismatch");
    }
}


int Dataset::getBatchSize() const {
    return batchSize;
}

Batch Dataset::getBatch(int batchidx) const {
    int startIdx = batchidx * batchSize;
    int endIdx = std::min(startIdx + batchSize, numSamples);
    int actualBatchSize = endIdx - startIdx;

    Batch b;
    b.batchX = X.middleRows(startIdx, actualBatchSize);
    b.batchY = Y.middleRows(startIdx, actualBatchSize);
    return b;
}

void Dataset::shuffleData(std::optional<unsigned int> seed) {
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 g(seed.has_value() ? seed.value() : std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::MatrixXf xShuffled(X.rows(), X.cols());
    Eigen::MatrixXf yShuffled(Y.rows(), Y.cols());

    for (int i = 0; i < indices.size(); ++i) {
        xShuffled.row(i) = X.row(indices[i]);
        yShuffled.row(i) = Y.row(indices[i]);
    }

    X = xShuffled;
    Y = yShuffled;
}

int Dataset::getNumSamples() const{
    return numSamples;
}
