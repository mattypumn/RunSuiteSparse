
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
  solve_economy_ = false;
  do_permutations_ = true;
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


size_t solve_counter = 0;
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
  LOG(INFO) << "FLoat solved: " << ++solve_counter;
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


size_t SparseSystemFloat::TimeQrDecomposition(
    std::vector<float>* QT_b, std::vector<Triplet>* R_triplets,
    std::vector<size_t>* permutation) {
  CHECK_NOTNULL(b_);
  CHECK_NOTNULL(A_);
  CHECK_NOTNULL(QT_b);
  CHECK_NOTNULL(R_triplets);
  CHECK_NOTNULL(permutation);

  cholmod_dense *C = nullptr;  // Q' * b.
  cholmod_sparse *R = nullptr;  // R decomposition.
  SuiteSparse_long *p = nullptr;  // Permutation vector.

  const auto start = std::chrono::high_resolution_clock::now();
  // Wrapper function.
//   SuiteSparseQR<float>(SPQR_ORDERING_FIXED, SPQR_DEFAULT_TOL, A_->ncol,
//                         A_, b_, &C, &R, &p, &cc_);
//   // Actual function call skipping the wrapper function.
  SuiteSparseQR<float>((do_permutations_) ?
                            SPQR_ORDERING_DEFAULT : SPQR_ORDERING_FIXED,
                        SPQR_DEFAULT_TOL,
                        (solve_economy_) ? 0 : A_->ncol,
                        0,                  // Solve C = Q' * B.
                        A_,
                        NULL, b_,           // b_sparse, b_dense.
                        NULL, &C,           // z_sparse, z_dense.
                        &R,                 // R-factor.
                        &p,                 // Permutation vector.
                        nullptr, nullptr, nullptr,  // Unused output.
                        &cc_);


  const auto stop = std::chrono::high_resolution_clock::now();
  const auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();
  LOG(INFO) << "Solve economy: " << solve_economy_;
  LOG(INFO) << "R size: (" << R->nrow << " x " << R->ncol << ")";
  LOG(INFO) << "C size: (" << C->nrow << " x " << C->ncol << ")";

  CHECK_EQ(b_->ncol, C->ncol);
  CHECK_EQ(A_->ncol, C->nrow);

  // Fill R triplets.
  if (R != nullptr) {
  CholmodSparseToTriplet(R, R_triplets);
  } else {
    LOG(WARNING) << "R is nullptr ?!";
  }

  // Fill QT_b.
  QT_b->clear();
  if (C != nullptr) {
    QT_b->reserve(C->nrow);
    for (size_t row_i = 0; row_i < C->nrow; ++row_i) {
      QT_b->push_back(static_cast<float *>(C->x)[row_i]);
    }
  }
  LOG(INFO) << "QtB vecotr size: " << QT_b->size();

  // Set permutation.
  permutation->clear();
  permutation->reserve(A_->ncol);
  for (size_t perm_i = 0; perm_i < A_->ncol; ++perm_i) {
    if (p == nullptr) {
      permutation->push_back(perm_i);
    } else {
      permutation->push_back(p[perm_i]);
    }
  }

  cholmod_l_free(A_->ncol, sizeof(SuiteSparse_long), p, &cc_);
  cholmod_l_free_dense(&C, &cc_);
  cholmod_l_free_sparse(&R, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;

  return time_ns;
}

void SparseSystemFloat::CholmodSparseToTriplet(
    cholmod_sparse* A, std::vector<Triplet>* triplets) {
  triplets->clear();
  cholmod_triplet *trip = cholmod_l_sparse_to_triplet(A, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  const size_t nnz = cholmod_l_nnz(A, &cc_);
  triplets->reserve(trip->nnz);
  for (size_t iter = 0; iter < nnz; ++iter) {
    triplets->emplace_back(static_cast<size_t *>(trip->i)[iter],
                           static_cast<size_t *>(trip->j)[iter],
                           static_cast<float *>(trip->x)[iter]);
  }

  cholmod_l_free_triplet(&trip, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
}

void SparseSystemFloat::SetRhs(const std::vector<float>& rhs) {
  CHECK_EQ(b_->nrow, rhs.size()) << " vector of incorrect size for system Ax=b";
  for (size_t val_i = 0; val_i < rhs.size(); ++val_i) {
    static_cast<float *>(b_->x)[val_i] = rhs[val_i];
  }
}

}  // namespace sparse_qr.
