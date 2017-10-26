#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <string>

#include <SuiteSparseQR.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

constexpr bool kLoadKouroshMatrix = true;

extern struct SuiteSparse_config_struct SuiteSparse_config;

template<typename T>
void ReadBinary(std::istream& is, std::vector<T>& v) {
  std::uint64_t sz;
  is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  v.resize(sz);
  is.read(reinterpret_cast<char*>(v.data()), sizeof(T) * sz);
}

void cholmod_error_handler(int status, const char *file,
        int line, const char *message) {
  LOG(ERROR) << "CHOLMOD ERROR: file: " << file << " line: " << line <<
    " status = " << status << " message: " << message;
}

int mprinter(const char* format, ...) {
  char buf[200];
  va_list args;
  va_start(args, format);
  vsprintf(buf, format, args);
  LOG(ERROR) << buf;
  return 0;
}

cholmod_sparse* LoadKouroshTransposeMatrix(
    const std::string& filename, cholmod_common* cc) {
  CHECK(cc->status == CHOLMOD_OK);

  std::ifstream file(filename);
  std::vector<std::uint64_t> rows, cols;
  std::vector<double> vals;
  size_t num_rows, num_cols, nnz;

  if (!file.is_open()) {
    LOG(FATAL) << "(ReadSparseMatrix) Unable to open file: " << filename;
  }

  file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
  file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
  file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

  ReadBinary(file, cols);
  ReadBinary(file, rows);
  ReadBinary(file, vals);

  CHECK_EQ(rows.size(), cols.size());
  CHECK_EQ(rows.size(), vals.size());
  CHECK_GT(rows.size(), 0);

  LOG(INFO) << "matrix file: " << filename;
  LOG(INFO) << "nnz: " << nnz;
  LOG(INFO) << "num_rows: " << num_rows;
  LOG(INFO) << "num_cols: " << num_cols;

  LOG(INFO) << "CHOLMOD_PATTERN " << CHOLMOD_PATTERN;
  LOG(INFO) << "Real code: " << CHOLMOD_REAL;
  LOG(INFO) << "CHOLMOD_ZOMPLEX " << CHOLMOD_ZOMPLEX;

  cholmod_triplet *trip = cholmod_allocate_triplet(num_rows, num_cols,
                                        vals.size(), 0, CHOLMOD_REAL, cc);
  CHECK(cc->status == CHOLMOD_OK) << "Error code: " << cc->status;

  CHECK_NOTNULL(trip);
  CHECK_NOTNULL(trip->i);
  CHECK_NOTNULL(trip->j);
  CHECK_NOTNULL(trip->x);

  for (size_t iter = 0; iter < vals.size(); ++iter) {
    CHECK_LT(cols[iter], num_cols);
    CHECK_LT(rows[iter], num_rows);
    static_cast<int *>(trip->i)[trip->nnz] = static_cast<int>(rows[iter]);
    static_cast<int *>(trip->j)[trip->nnz] = static_cast<int>(cols[iter]);
    static_cast<float *>(trip->x)[trip->nnz] = static_cast<float>(vals[iter]);
    ++trip->nnz;
    LOG(INFO) << iter;
  }
  cholmod_sparse* A = cholmod_triplet_to_sparse(trip, 0, cc);
  cholmod_l_free_triplet(&trip, cc);
  return A;
}



int main (int argc, char **argv)
{
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
  cc->error_handler = &cholmod_error_handler;
  SuiteSparse_config.printf_func = &mprinter;
  LOG(INFO) << "TRYCATCH = " << cc->try_catch;
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
  if (kLoadKouroshMatrix) {
    fclose(matrix_fid);
    A_sparse = LoadKouroshTransposeMatrix(matrix_file, cc);

    if (A_sparse == nullptr) {
      LOG(FATAL) << "Could not load sparse matrix.";
    } else {
      LOG(ERROR) << "Loaded sparse matrix";
    }
  } else {
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
    cholmod_l_free_dense(&A_dense, cc);
  }

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
  cholmod_l_free_dense (&Residual, cc) ;
  cholmod_l_free_sparse (&A_sparse, cc) ;
  cholmod_l_free_dense (&X, cc) ;
  cholmod_l_free_dense (&B, cc) ;
  cholmod_l_finish (cc) ;
  return (0) ;
}
