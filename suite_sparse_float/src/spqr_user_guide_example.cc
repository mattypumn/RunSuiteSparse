#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <string>

#include <SuiteSparseQR.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

int main (int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  // Usage.
  if (argc != 3) {
    LOG(ERROR) << "Usage: <" << argv[0] << "> <matrix-file> <result-file>";
    return -1;
  }

  // Build needed data objects.
  cholmod_common Common, *cc;
  cholmod_sparse *A_sparse;
  cholmod_dense *X, *B, *Residual;
  float rnorm, one [2] = {1,0}, minusone [2] = {-1,0};
  // start CHOLMOD
  cc = &Common;
  cholmod_l_start (cc);
  cc->try_catch = 0;

  // Load and check read and write file.
  const std::string matrix_file = argv[1];
  const std::string result_file = argv[2];
  FILE* matrix_fid = fopen(matrix_file.c_str(), "r");
  FILE* result_fid = fopen(result_file.c_str(), "w");

  // Safety check files.
  CHECK_NOTNULL(matrix_fid);
  CHECK_NOTNULL(result_fid);

  // Load matrix.
  cholmod_dense *A_dense;
  A_dense = (cholmod_dense *) cholmod_l_read_dense (matrix_fid, cc);
  A_sparse = cholmod_l_dense_to_sparse(A_dense, 1, cc);
  fclose(matrix_fid);

  if (A_dense == nullptr) {
    LOG(FATAL) << "Could not load matrix: " << matrix_file;
  } else {
    LOG(ERROR) << "Loaded matrix: " << matrix_file;
  }
  if (A_sparse == nullptr) {
    LOG(FATAL) << "Could not convert matrix to sparse.";
  } else {
    LOG(ERROR) << "Converted Matrix to sparse";
  }
  const float Adense_norm = cholmod_l_norm_dense(A_dense, 2, cc);
  LOG(INFO) << "A_dense norm: " << Adense_norm;

  const float Asparse_norm = cholmod_l_norm_sparse(A_sparse, 2, cc);
  LOG(INFO) << "A_sparse norm: " << Asparse_norm;


  // Create B for AX = B.   B = ones (size (A,1),1)
  B = cholmod_l_ones (A_sparse->nrow, 1, A_sparse->xtype, cc);

  // Solve X = A\B.
  X = SuiteSparseQR <float> (A_sparse, B, cc) ;
  LOG(ERROR) << "Status: " << cc->status <<
                " (CHOLMOD_OK = " << CHOLMOD_OK << ")";

  // Save the resulting X.
  cholmod_l_write_dense(result_fid, X, NULL, cc);
  fclose(result_fid);

  // Calculate the 2norm of the residual    norm (B-A*X).
  Residual = cholmod_l_copy_dense (B, cc) ;
  cholmod_l_sdmult (A_sparse, 0, minusone, one, X, Residual, cc) ;
  rnorm = cholmod_l_norm_dense (Residual, 2, cc);

  // Print results.
  LOG(ERROR) << "2-norm of residual: " << rnorm;
  LOG(ERROR) << "rank: " << cc->SPQR_istat [4];

  // Free everything and finish CHOLMOD.
  cholmod_l_free_dense(&A_dense, cc);
  cholmod_l_free_dense (&Residual, cc) ;
  cholmod_l_free_sparse (&A_sparse, cc) ;
  cholmod_l_free_dense (&X, cc) ;
  cholmod_l_free_dense (&B, cc) ;
  cholmod_l_finish (cc) ;
  return (0) ;
}
