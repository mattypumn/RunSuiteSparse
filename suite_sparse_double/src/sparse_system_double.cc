#include "../include/sparse_qr/sparse_system_double.h"

#include <chrono>
#include <glog/logging.h>
#include <SuiteSparseQR.hpp>
#include <spqr.hpp>
#include <tuple>
#include <vector>


namespace sparse_qr{

void LogError(int status, const char *file,
        int line, const char *message) {
  LOG(INFO) << "CHOLMOD ERROR status: " << status << " file: " << file <<
    " line: " << line << " message: " << message;
}

SparseSystemDouble::SparseSystemDouble() : num_threads_(0), num_cores_(0) {
  LOG(FATAL) << "Constructor Not implemented.";
}


SparseSystemDouble::SparseSystemDouble(
    const size_t& rows, const size_t& cols,
    const std::vector<Triplet>& vals,
    const size_t& num_threads, const size_t& num_cores)
  : num_threads_(num_threads), num_cores_(num_cores) {
  cholmod_start(&cc_);
  cc_.itype = CHOLMOD_LONG;
  cc_.error_handler = &LogError;
  cc_.try_catch = 0;
  cc_.SPQR_nthreads = static_cast<int>(num_threads_);
  cc_.SPQR_grain = 2 * static_cast<int>(num_cores_);

  b_ = cholmod_l_ones(rows, 1, CHOLMOD_REAL, &cc_);

  cholmod_triplet *trip = cholmod_l_allocate_triplet(rows, cols, vals.size(),
                                                   0, CHOLMOD_REAL, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  CHECK_EQ(trip->nnz, 0);
  for (const auto& entry : vals) {
    static_cast<long *>(trip->i)[trip->nnz] = std::get<0>(entry);
    static_cast<long *>(trip->j)[trip->nnz] = std::get<1>(entry);
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
  cholmod_dense* x = SuiteSparseQR<double>(A_, b_, &cc_);
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
//   LOG(INFO) << "solved: " << ++solve_counter << " time (s): " <<
//                 time_ns * kNanoToSeconds;
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

void SparseSystemDouble::CholmodSparseToTriplet(
    cholmod_sparse* M, std::vector<SparseSystemDouble::Triplet>* triplets) {
  triplets->clear();
  cholmod_triplet *trip = cholmod_l_sparse_to_triplet(M, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  triplets->reserve(trip->nnz);
  for (size_t iter = 0; iter < trip->nnz; ++iter) {
    triplets->emplace_back(static_cast<size_t *>(trip->i)[iter],
                           static_cast<size_t *>(trip->j)[iter],
                           static_cast<double *>(trip->x)[iter]);
  }
  cholmod_l_free_triplet(&trip, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
}

size_t SparseSystemDouble::TimeQrDecomposition(
    std::vector<double>* QT_b, std::vector<Triplet>* R_triplets,
    std::vector<size_t>* permutation) {
  if (b_ == nullptr) {
    b_ = cholmod_l_ones(A_->nrow, 1, A_->xtype, &cc_);
  }
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;

  cholmod_dense *Z;  // Q' * b.
  cholmod_sparse *R;  // R decomposition.
  SuiteSparse_long *E;  // Permutation vector.

  const auto start = std::chrono::high_resolution_clock::now();
//   // To decompose without solving for Q' * b.
//   SuiteSparseQR<double>(SPQR_ORDERING_DFAULT, SPQR_DEFAULT_TOL, A_->ncol, 0,
//               A_, NULL, NULL, NULL, NULL, &R, &E, NULL, NULL, NULL, &cc_);
  // To decompose and solve for Z = Q'*b.
  SuiteSparseQR<double>(SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL,
                        A_->ncol,           // Rows of Z and R to return.
                        0,                  // Z = C = Q'*b.
                        A_, NULL, b_,       // Input.
                        NULL, &Z, &R, &E, NULL, NULL, NULL, &cc_); // Output.
  const auto stop = std::chrono::high_resolution_clock::now();
  const auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();
  CHECK_EQ(b_->ncol, Z->ncol);
  CHECK_EQ(A_->ncol, Z->nrow);

  // Fill R triplets.
  CholmodSparseToTriplet(R, R_triplets);

  // Fill QT_b.
  QT_b->clear();
  QT_b->reserve(Z->nrow);
  for (size_t row_i = 0; row_i < Z->nrow; ++row_i) {
    QT_b->push_back(static_cast<double *>(Z->x)[row_i]);
  }

  // Set permutation.
  permutation->clear();
  permutation->reserve(A_->ncol);
  for (size_t perm_i = 0; perm_i < A_->ncol; ++perm_i) {
    permutation->push_back(E[perm_i]);
  }

  cholmod_l_free(A_->ncol, sizeof(SuiteSparse_long), E, &cc_);
  cholmod_l_free_dense(&Z, &cc_);
  cholmod_l_free_sparse(&R, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;

  return time_ns;
}



}  // namespace sparse_qr.


