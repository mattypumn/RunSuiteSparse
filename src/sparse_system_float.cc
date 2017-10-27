
#include "../include/sparse_qr/sparse_system_float.h"

#include <chrono>
#include <glog/logging.h>
#include <SuiteSparseQR.hpp>


namespace sparse_qr {

void LogError(int status, const char *file,
        int line, const char *message) {
  LOG(FATAL) << "CHOLMOD ERROR status: " << status << " file: " << file <<
    " line: " << line << " message: " << message;
}

SparseSystemFloat::SparseSystemFloat(){
  LOG(FATAL) << "Constructor Should NOT be used.";
}

SparseSystemFloat::SparseSystemFloat(
    const size_t& rows, const size_t& cols,
    const std::vector<Triplet>& vals){
  cholmod_start(&cc_);
  cc_.error_handler = &LogError;
  cc_.itype = CHOLMOD_LONG;
  CHECK_EQ(cc_.status, CHOLMOD_OK) << "cholmod status: " << cc_.status;
  cholmod_triplet *trip = cholmod_l_allocate_triplet(rows, cols, vals.size(),
                                                   0, CHOLMOD_REAL, &cc_);

  for (const auto& entry : vals) {
    static_cast<size_t *>(trip->i)[trip->nnz] = std::get<0>(entry);
    static_cast<size_t *>(trip->j)[trip->nnz] = std::get<1>(entry);
    static_cast<float *>(trip->x)[trip->nnz] = std::get<2>(entry);
    ++trip->nnz;
  }

  A_ = cholmod_l_triplet_to_sparse(trip, 0, &cc_);
  b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);

  cholmod_l_free_triplet(&trip, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << "cholmod status: " << cc_.status;
}

SparseSystemFloat::~SparseSystemFloat() {
  cholmod_l_free_dense(&b_, &cc_);
  cholmod_l_free_sparse(&A_, &cc_);
  cholmod_finish(&cc_);
}


void SparseSystemFloat::SetDimensions(const size_t& rows, const size_t& cols) {
  LOG(FATAL) << "NOT YET IMPLMENTED";
}

void SparseSystemFloat::SetWithTuples(const std::vector<Triplet>& vals) {
  LOG(FATAL) << "NOT YET IMPLEMENTED";
}

void SparseSystemFloat::SetRhs(std::vector<float> rhs) {
  LOG(FATAL) << "Not yet implemented";
}

size_t SparseSystemFloat::TimeSolve(float* residual_norm) {
  CHECK_NOTNULL(A_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << "cholmod status: " << cc_.status;
  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
    CHECK_EQ(cc_.status, CHOLMOD_OK) << "cholmod status: " << cc_.status;
  }
  // Time the QR Solve.
  const auto start = std::chrono::high_resolution_clock::now();
  cholmod_dense* x = SolveQR();
  const auto stop = std::chrono::high_resolution_clock::now();

  if (residual_norm != nullptr) {
    float one[2] = {1,0}, minusone[2] = {-1,0};
    cholmod_dense* res = cholmod_l_copy_dense (b_, &cc_);
    // res = one * (A * x) + minusone * res;
    cholmod_l_sdmult(A_, 0, minusone, one, x, res, &cc_);
    (*residual_norm) = cholmod_l_norm_dense(res, 2, &cc_);
    cholmod_l_free_dense(&res, &cc_);
  }

  const size_t time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();
  cholmod_l_free_dense(&x, &cc_);

  CHECK_EQ(cc_.status, CHOLMOD_OK) << "cholmod status: " << cc_.status;
  return time_ns;
}

void SparseSystemFloat::TimeSolveN(
    const size_t& n_solves, std::vector<size_t>* times_ns,
    std::vector<float>* residual_norms) {
  CHECK_NOTNULL(times_ns);\
  CHECK_NOTNULL(A_);

  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
    CHECK_EQ(cc_.status, CHOLMOD_OK) << "cholmod status: " << cc_.status;
    CHECK_NOTNULL(b_);
  }

  size_t prev_norms_count = 0;
  times_ns->reserve(n_solves);
  if (residual_norms != nullptr) {
    prev_norms_count = residual_norms->size();
    residual_norms->resize(n_solves + prev_norms_count);
  }

  for (size_t i = 0; i < n_solves; ++i) {
    if (residual_norms == nullptr) {
      times_ns->push_back(TimeSolve());
    } else {
      float* res = &(residual_norms->at(prev_norms_count + i));
      times_ns->push_back(TimeSolve(res));
    }
  }
}

size_t SparseSystemFloat::SystemSolve(
    const std::vector<float>& rhs, std::vector<float>* x) {
  LOG(FATAL) << "NOT YET IMPLEMENTED";
//   cholmod_dense* x_solve = SuiteSparseQR <float> (A_, b_, &cc_);
//   cholmod_l_free_dense(x_solve, cc_);
  return 0;
}
cholmod_dense* SparseSystemFloat::SolveQR() {
  return SuiteSparseQR <float> (A_, b_, &cc_);
}
//
// cholmod_dense* SparseSystemFloat::sparse_qr(
//     cholmod_sparse* A, cholmod_dense* b, cholmod_common* cc) {
//   return SuiteSparseQR <float> (A, b, cc);
// }

}  // namespace sparse_qr.

