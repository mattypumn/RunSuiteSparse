#include <algorithm>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <vector>

 #include <glog/logging.h>
#include <gflags/gflags.h>

#include "../include/sparse_qr/sparse_system_float.h"

constexpr size_t kNumSolves = 100;
const std::string times_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/float_times.txt";
const std::string residuals_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/float_res.txt";

extern struct SuiteSparse_config_struct SuiteSparse_config;

typedef sparse_qr::SparseSystemFloat::Triplet triplet_f;

int LogError(const char* format, ...) {
  char buf[1027];
  va_list args;
  va_start(args, format);
  vsprintf(buf, format, args);
  LOG(ERROR) << buf;
  return 0;
}


template<typename T>
void ReadBinary(std::istream& is, std::vector<T>& v) {
  std::uint64_t sz;
  is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  v.resize(sz);
  is.read(reinterpret_cast<char*>(v.data()), sizeof(T) * sz);
}

void ReadSparseMatrix(
    const std::string& filename, size_t* num_rows, size_t* num_cols,
    std::vector<triplet_f>* entries_float) {
  std::ifstream file(filename);
  std::vector<std::uint64_t> rows, cols;
  std::vector<double> vals;
  size_t nnz;

  if (!file.is_open()) {
    LOG(FATAL) << "(ReadSparseMatrix) Unable to create file: " << filename;
  }
  file.read(reinterpret_cast<char*>(num_cols), sizeof(*num_cols));
  file.read(reinterpret_cast<char*>(num_rows), sizeof(*num_rows));
  file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));


  ReadBinary(file, cols);
  ReadBinary(file, rows);
  ReadBinary(file, vals);

  CHECK_EQ(rows.size(), cols.size());
  CHECK_EQ(rows.size(), vals.size());

  for (size_t iter = 0; iter < vals.size(); ++iter) {
    CHECK_LT(cols[iter], *num_cols);
    CHECK_LT(rows[iter], *num_rows);
    entries_float->emplace_back(rows[iter], cols[iter],
                                static_cast<float>(vals[iter]));
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  SuiteSparse_config.printf_func = &LogError;

  if (argc < 2) {
    LOG(ERROR) << "Usage: <" << argv[0] << "> <binary-matrix-file-1> ...";
    return -1;
  }

  std::vector<std::string> matrix_files;
  for (int i = 1; i < argc; ++i) {
    matrix_files.push_back(argv[i]);
  }
  CHECK_GT(matrix_files.size(), 0) << "Must have more than one matrix file";
  LOG(INFO) << "Found Matrix files. Count: " << matrix_files.size();

  std::vector<size_t> float_times_ns;
  std::vector<float> residual_norms;
  std::random_shuffle(matrix_files.begin(), matrix_files.end());
  for (const std::string& file : matrix_files) {
    size_t num_cols, num_rows;

    /* Floats */
    std::vector<triplet_f> triplets_f;
    ReadSparseMatrix(file, &num_rows, &num_cols, &triplets_f);

    sparse_qr::SparseSystemFloat system_f(num_rows, num_cols, triplets_f);


    system_f.TimeSolveN(kNumSolves, &float_times_ns, &residual_norms);
  }

  CHECK_GT(float_times_ns.size(), 0);
  CHECK_GT(residual_norms.size(), 0);

  std::ofstream tfile(times_file);
  for (const auto& time : float_times_ns) {
    tfile << time << std::endl;
  }
  tfile.close();

  std::ofstream rfile(residuals_file);
  for (const auto& res : residual_norms) {
    rfile << std::setprecision(15) << res << std::endl;
//     LOG(INFO) << "res: " << res;
  }
  rfile.close();
}
