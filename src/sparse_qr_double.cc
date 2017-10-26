#include "../include/sparse_qr/sparse_qr_double.h"

#include <chrono>
#include <glog/logging.h>
#include <SuiteSparseQR.hpp>
#include <tuple>
#include <vector>

namespace sparse_qr{

SparseSystemDouble::SparseSystemDouble(){
  LOG(FATAL) << "Constructor Not implemented.";
}

SparseSystemDouble::SparseSystemDouble(
    const size_t& rows, const size_t& cols,
    const std::vector<Triplet>& vals){
  cholmod_start(&cc_);
  cholmod_triplet *trip = cholmod_allocate_triplet(rows, cols, vals.size(),
                                                   0, CHOLMOD_REAL, &cc_);
  for (const auto& entry : vals) {
    static_cast<int *>(trip->i)[trip->nnz] =
        static_cast<int>(std::get<0>(entry));
    static_cast<int *>(trip->j)[trip->nnz] =
        static_cast<int>(std::get<1>(entry));
    static_cast<double *>(trip->x)[trip->nnz] = std::get<2>(entry);
    ++trip->nnz;
  }
  A_ = cholmod_triplet_to_sparse(trip, 0, &cc_);
  b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);

  cholmod_free_triplet(&trip, &cc_);
}

SparseSystemDouble::~SparseSystemDouble() {
  cholmod_l_free_dense(&b_, &cc_);
  cholmod_l_free_sparse(&A_, &cc_);
  cholmod_finish(&cc_);
}
//
void SparseSystemDouble::SetDimensions(const size_t& rows, const size_t& cols) {
  LOG(FATAL) << "NOT YET IMPLMENTED";
}

void SparseSystemDouble::SetWithTuples(const std::vector<Triplet>& vals) {
  LOG(FATAL) << "NOT YET IMPLEMENTED";
}

void SparseSystemDouble::SetRhs(std::vector<double> rhs) {
  LOG(FATAL) << "Not yet implemented";
}

size_t SparseSystemDouble::TimeSolve() {
  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
  }
  const auto start = std::chrono::high_resolution_clock::now();
  cholmod_dense* x = sparse_qr();
  const auto stop = std::chrono::high_resolution_clock::now();
  const size_t time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();
  cholmod_l_free_dense(&x, &cc_);
  return time_ns;
}

void SparseSystemDouble::TimeSolveN(
    const size_t& n_solves, std::vector<size_t>* times_ns) {
  CHECK_NOTNULL(times_ns);
  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
  }
  times_ns->reserve(n_solves);
  for (size_t i = 0; i < n_solves; ++i) {
    times_ns->push_back(TimeSolve());
  }
}

size_t SparseSystemDouble::SystemSolve(
    const std::vector<double>& rhs, std::vector<double>* x) {
  LOG(FATAL) << "NOT YET IMPLEMENTED";
  return 0;
}

cholmod_dense* SparseSystemDouble::sparse_qr() {
  return SuiteSparseQR <double> (A_, b_, &cc_);
}
//
// cholmod_dense* sparse_qr_double(
//     cholmod_sparse* A, cholmod_dense* b, cholmod_common* cc) {
//   cholmod_dense* x = SuiteSparseQR <double> (A, b, cc);
//   return x;
}  // namespace sparse_qr.

