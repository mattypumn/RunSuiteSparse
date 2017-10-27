#include "../include/sparse_qr/sparse_system_double.h"

#include <chrono>
#include <glog/logging.h>
#include <SuiteSparseQR.hpp>
#include <tuple>
#include <vector>

void error_handler(int status, const char *file,
        int line, const char *message) {
  LOG(INFO) << "CHOLMOD ERROR status: " << status << " file: " << file <<
    " line: " << line << " message: " << message;
}

namespace sparse_qr{

SparseSystemDouble::SparseSystemDouble(){
  LOG(FATAL) << "Constructor Not implemented.";
}

SparseSystemDouble::SparseSystemDouble(
    const size_t& rows, const size_t& cols,
    const std::vector<Triplet>& vals){
  cholmod_start(&cc_);
  cc_.itype = CHOLMOD_LONG;
  cc_.error_handler = &error_handler;
  cc_.try_catch = 0;

  b_ = cholmod_l_ones(rows, 1, CHOLMOD_REAL, &cc_);

  cholmod_triplet *trip = cholmod_l_allocate_triplet(rows, cols, vals.size(),
                                                   0, CHOLMOD_REAL, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  CHECK_EQ(trip->nnz, 0);
  for (const auto& entry : vals) {
    static_cast<long *>(trip->i)[trip->nnz] =
        static_cast<int>(std::get<0>(entry));
    static_cast<long *>(trip->j)[trip->nnz] =
        static_cast<int>(std::get<1>(entry));
    static_cast<double *>(trip->x)[trip->nnz] = std::get<2>(entry);
    ++trip->nnz;
  }
  A_ = cholmod_l_triplet_to_sparse(trip, 0, &cc_);
//   LOG(INFO) << "NUM ROWS = " << A_->nrow;
//   b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
//   CHECK_EQ(cc_.status, CHOLMOD_OK);

  cholmod_l_free_triplet(&trip, &cc_);

  CHECK(cc_.status == CHOLMOD_OK) << " cholmod status: " << cc_.status;
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

size_t solve_counter = 0;
size_t SparseSystemDouble::TimeSolve(double* residual_norm) {
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
    CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  }
  const auto start = std::chrono::high_resolution_clock::now();
  cholmod_dense* x = sparse_qr();
  const auto stop = std::chrono::high_resolution_clock::now();

  // Calculate and save residual norm.
  if (residual_norm != nullptr) {
    double one[2] = {1,0}, minusone[2] = {-1,0};
    cholmod_dense* res = cholmod_l_copy_dense (b_, &cc_);
    // res = one * (A * x) + minusone * res;
    cholmod_l_sdmult(A_, 0, minusone, one, x, res, &cc_);
    (*residual_norm) = cholmod_l_norm_dense(res, 2, &cc_);
    cholmod_l_free_dense(&res, &cc_);
  }

  const size_t time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();
  cholmod_l_free_dense(&x, &cc_);
  LOG(INFO) << "solved: " << ++solve_counter;
  return time_ns;
}

void SparseSystemDouble::TimeSolveN(
    const size_t& n_solves, std::vector<size_t>* times_ns,
    std::vector<double>* residual_norms) {
  const bool save_residual = (residual_norms != nullptr);
  CHECK_NOTNULL(times_ns);
  CHECK(cc_.status == CHOLMOD_OK) << " cholmod status: " << cc_.status;
  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
  }
  times_ns->reserve(n_solves);
  size_t prev_norms_count = 0;
  if (save_residual) {
    prev_norms_count = residual_norms->size();
    residual_norms->resize(prev_norms_count + n_solves);
  }

  // Solve n-times.
  if (save_residual) {
    for (size_t i = 0; i < n_solves; ++i) {
      double* res_ptr = &(residual_norms->at(prev_norms_count + i));
      times_ns->push_back(TimeSolve(res_ptr));
    }
  } else {
    for (size_t i = 0; i < n_solves; ++i) {
      times_ns->push_back(TimeSolve());
    }
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
// cholmod_l_dense* sparse_qr_double(
//     cholmod_l_sparse* A, cholmod_l_dense* b, cholmod_l_common* cc) {
//   cholmod_l_dense* x = SuiteSparseQR <double> (A, b, cc);
//   return x;
}  // namespace sparse_qr.

