#include "../include/sparse_qr/sparse_system_double.h"

#include <chrono>
#include <tuple>
#include <vector>

#include <cs.h>
#include <glog/logging.h>
#include <SuiteSparseQR.hpp>
#include <spqr.hpp>

namespace sparse_qr{

// WARNING
// NOTE
// TODO
// WTF

constexpr double kSpqrSmall = 1e6;
constexpr int kSpqrTryCatch = 0;
constexpr int kSpqrShrink = 1;
constexpr int kCoresToGrain = 2;
constexpr double kMetisMemory = 0.0;


void LogError(int status, const char *file,
        int line, const char *message) {
  LOG(ERROR) << "CHOLMOD ERROR status: " << status << " file: " << file <<
    " line: " << line << " message: " << message;
}

SparseSystemDouble::SparseSystemDouble() : num_threads_(0), num_cores_(0) {
  LOG(FATAL) << "Constructor Purposely hidden.";
}

// TODO(mpoulter) A SystemFactory should probably be implemented rather than,
// this constructor.
SparseSystemDouble::SparseSystemDouble(
    const size_t& rows, const size_t& cols,
    const std::vector<Triplet>& A_vals,
    const size_t& num_threads, const size_t& num_cores)
  : num_threads_(num_threads), num_cores_(num_cores) {
  cholmod_start(&cc_);
  cc_.itype = CHOLMOD_LONG;
  cc_.error_handler = &LogError;
  cc_.try_catch = kSpqrTryCatch;
  cc_.SPQR_nthreads = static_cast<int>(num_threads_);
  cc_.SPQR_grain = (num_cores == 1) ? num_cores :
                   kCoresToGrain * static_cast<int>(num_cores_);
  // More options to mimic Matlab implementation of SPQR.
  cc_.metis_memory = kMetisMemory;
  cc_.SPQR_shrink = kSpqrShrink;
  cc_.SPQR_small = kSpqrSmall;

  b_ = cholmod_l_ones(rows, 1, CHOLMOD_REAL, &cc_);

  cholmod_triplet *trip = cholmod_l_allocate_triplet(rows, cols, A_vals.size(),
                                                   0, CHOLMOD_REAL, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  CHECK_EQ(trip->nnz, 0);
  for (const auto& entry : A_vals) {
    static_cast<long *>(trip->i)[trip->nnz] = std::get<0>(entry);
    static_cast<long *>(trip->j)[trip->nnz] = std::get<1>(entry);
    static_cast<double *>(trip->x)[trip->nnz] = std::get<2>(entry);
    ++trip->nnz;
  }
  A_ = cholmod_l_triplet_to_sparse(trip, 0, &cc_);

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

void SparseSystemDouble::SetRhs(const std::vector<double>& rhs) {
  CHECK_EQ(b_->ncol, rhs.size()) << " vector of incorrect size for system Ax=b";
  for (size_t val_i = 0; val_i < rhs.size(); ++val_i) {
    static_cast<double *>(b_->x)[val_i] = rhs[val_i];
  }
}

size_t SparseSystemDouble::TimeSolve(
    double* residual_norm, std::vector<double>* x_solve) {
  CHECK_NOTNULL(b_);
  CHECK_NOTNULL(A_);

  // Time solve.
  const auto start = std::chrono::high_resolution_clock::now();
  cholmod_dense* x = SuiteSparseQR<double>(A_, b_, &cc_);
  const auto stop = std::chrono::high_resolution_clock::now();
  const size_t time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();

  // Calculate and save residual norm.
  if (residual_norm != nullptr) {
    double one[2] = {1,0}, minusone[2] = {-1,0};
    cholmod_dense* res = cholmod_l_copy_dense (b_, &cc_);
    // res = one * (A * x) + minusone * res;
    cholmod_l_sdmult(A_, 0, minusone, one, x, res, &cc_);
    (*residual_norm) = cholmod_l_norm_dense(res, 2, &cc_);
    cholmod_l_free_dense(&res, &cc_);
  }

  // Save the solution vector.
  if (x_solve != nullptr) {
    x_solve->clear();
    x_solve->reserve(x->nrow);
    for (size_t val_i = 0; val_i < x->nrow; ++val_i) {
      x_solve->push_back(static_cast<double *>(x->x)[val_i]);
    }
  }

  // Free resources.
  cholmod_l_free_dense(&x, &cc_);
  CHECK(cc_.status == CHOLMOD_OK) << " cholmod status: " << cc_.status;

  return time_ns;
}

void SparseSystemDouble::TimeSolveN(
    const size_t& n_solves, std::vector<size_t>* times_ns,
    std::vector<double>* residual_norms) {
  const bool save_residual = (residual_norms != nullptr);
  CHECK_NOTNULL(times_ns);

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
    cholmod_sparse* A, std::vector<SparseSystemDouble::Triplet>* triplets) {

  triplets->clear();
  cholmod_triplet *trip = cholmod_l_sparse_to_triplet(A, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
  const size_t nnz = cholmod_l_nnz(A, &cc_);
  LOG(INFO) << "actual nnz: " << nnz;
  LOG(INFO) << "prev reported nnz: " << trip->nnz;
  triplets->reserve(trip->nnz);
  for (size_t iter = 0; iter < nnz; ++iter) {
    triplets->emplace_back(static_cast<size_t *>(trip->i)[iter],
                           static_cast<size_t *>(trip->j)[iter],
                           static_cast<double *>(trip->x)[iter]);
  }

  cholmod_l_free_triplet(&trip, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;
}

// cholmod_dense* rsolve(cholmod_sparse* R, cholmod_dense* b, SuiteSparse_long p,
//     cholmod_common* cc) {
//   LOG(FATAL) << "Not implemented.";
//   return nullptr;
// }

size_t SparseSystemDouble::TimeQrDecomposition(
    std::vector<double>* QT_b, std::vector<Triplet>* R_triplets,
    std::vector<size_t>* permutation) {
  CHECK_NOTNULL(b_);
  CHECK_NOTNULL(A_);

  cholmod_dense *C = nullptr;  // Q' * b.
  cholmod_sparse *R = nullptr;  // R decomposition.
  SuiteSparse_long *p = nullptr;  // Permutation vector.

  // Sanity check to ensure B = 1-vector.
  for (int64_t i = 0; i < b_->ncol; ++i) {
    CHECK_EQ(static_cast<double *>(b_->x)[i], 1.);
  }

  const auto start = std::chrono::high_resolution_clock::now();
  // Wrapper function.
  SuiteSparseQR<double>(SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A_->ncol,
                        A_, b_, &C, &R, &p, &cc_);
//   // Actual function call skipping the wrapper function.
//   SuiteSparseQR<double>(SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL,
//                         A_->ncol,           // Rows of Z and R to return.
//                         0,                  // C = Q'*b.
//                         A_, NULL, b_,       // Input.
//                         NULL, &C, &R, &p, NULL, NULL, NULL, &cc_); // Output.
  const auto stop = std::chrono::high_resolution_clock::now();
  const auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start).count();
  LOG(INFO) << "R nnnz: " << R->nz;
  LOG(INFO) << "R max_nnnz: " << R->nzmax;

// TESTING Solving E*(R\(Q'*b)).
// NOTE  Could not find a handy way to solve for x = E*(R \(Q'*b)).
//   if (true && "TESTING") {
//     double one [2] = {1,0},
//            minusone [2] = {-1,0};
//     cholmod_factor *L = cholmod_l_allocate_factor(R->nrow, &cc_);
//     L->xtype = CHOLMOD_REAL;
//     cholmod_factorize(R, L, &cc_);
//     LOG(INFO) << "L xtype: " << L->xtype << " real: "  << CHOLMOD_REAL;
//     cholmod_dense *y = cholmod_l_solve(CHOLMOD_A, L, C, &cc_);   // Solve RE'*x = Q'*b;
//
//
//     cholmod_dense *x = cholmod_l_allocate_dense(y->nrow, y->ncol, y->ncol,
//                                                 CHOLMOD_REAL, &cc_);
//
//     for (int64_t y_idx = 0; y_idx < y->ncol; ++y_idx) {
//       size_t x_idx = E[y_idx];
//       static_cast<double *>(x->x)[x_idx] = static_cast<double *>(y->x)[y_idx];
//     }
//
//     cholmod_dense *res = cholmod_l_copy_dense (b_, &cc_) ;
//     cholmod_l_sdmult (A_, 0, minusone, one, x, res, &cc_) ;
//     const double rnorm = cholmod_l_norm_dense (res, 2, &cc_);
//     LOG(INFO) << "Residual norm from R*E*x=Q'*b: " << rnorm;
//     cholmod_l_free_dense(&x, &cc_);
//     cholmod_l_free_dense(&y, &cc_);
//     cholmod_l_free_dense(&res, &cc_);
//     cholmod_l_free_factor(&L, &cc_);
//   }

// TESTING x = SuiteSparseQR.
// NOTE  This shows that solving for x is working correctly.
//   if (true && "TESTING") {  // Test with a direct solve of x.
//     double one [2] = {1,0},
//            minusone [2] = {-1,0};
//     cholmod_dense *x;  // Q' * b.
//     SuiteSparseQR<double>(SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL,
//                           A_->ncol,           // Rows of Z and R to return.
//                           2,                  // C = Q'*b.
//                           A_, NULL, b_,       // Input.
//                           NULL, &x, &R, NULL, NULL, NULL, NULL, &cc_); // Output.
//     cholmod_dense *res = cholmod_l_copy_dense (b_, &cc_) ;
//     cholmod_l_sdmult (A_, 0, minusone, one, x, res, &cc_) ;
//     const double rnorm = cholmod_l_norm_dense (res, 2, &cc_);
//     LOG(INFO) << "Residual norm from solve: " << rnorm;
//     cholmod_l_free_dense(&x, &cc_);
//     cholmod_l_free_dense(&res, &cc_);
//   }



  CHECK_EQ(b_->ncol, C->ncol);
  CHECK_EQ(A_->ncol, C->nrow);

  // Fill R triplets.
  // WARNING check kSuiteSparseIsReturningTransposeForSomeDumbassReason.
  CholmodSparseToTriplet(R, R_triplets);

  // Fill QT_b.
  QT_b->clear();
  QT_b->reserve(C->nrow);
  for (size_t row_i = 0; row_i < C->nrow; ++row_i) {
    QT_b->push_back(static_cast<double *>(C->x)[row_i]);
  }

  // Set permutation.
  permutation->clear();
  permutation->reserve(A_->ncol);
  for (size_t perm_i = 0; perm_i < A_->ncol; ++perm_i) {
    permutation->push_back(p[perm_i]);
  }

  cholmod_l_free(A_->ncol, sizeof(SuiteSparse_long), p, &cc_);
  cholmod_l_free_dense(&C, &cc_);
  cholmod_l_free_sparse(&R, &cc_);
  CHECK_EQ(cc_.status, CHOLMOD_OK) << " cholmod status: " << cc_.status;

  return time_ns;
}



}  // namespace sparse_qr.


