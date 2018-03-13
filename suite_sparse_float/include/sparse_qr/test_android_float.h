#ifndef TEST_ANDROID_FLAOT_H
#define TEST_ANDROID_FLAOT_H

#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "../include/sparse_qr/sparse_system_float.h"

namespace android_suite_sparse {

std::pair<double, double> GetTimeStats(std::vector<double>& times_seconds);

void EigenSparseToTriplets(
    const Eigen::SparseMatrix<double>& S,
    std::vector<SparseSystemTriplet>* triplets);

void TripletsToEigenSparse(
    const std::vector<SparseSystemTriplet>& triplets,
    const size_t& rows, const size_t& cols,
    Eigen::SparseMatrix<double>* S);

std::pair<double, double> TimeQR(const std::string sparse_filepath);

}  // namespace android_suite_sparses

#endif TEST_ANDROID_FLAOT_H