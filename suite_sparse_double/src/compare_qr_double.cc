#include <algorithm>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "../include/sparse_qr/sparse_system_double.h"
#include <glog/logging.h>
#include <gflags/gflags.h>

constexpr size_t kNumSolves = 100;
const std::string time_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/double_times.txt";
const std::string res_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/double_res.txt";

typedef sparse_qr::SparseSystemDouble::Triplet triplet_d;

template<typename T>
void ReadBinary(std::istream& is, std::vector<T>& v) {
  std::uint64_t sz;
  is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  v.resize(sz);
  is.read(reinterpret_cast<char*>(v.data()), sizeof(T) * sz);
}

void ReadSparseMatrix(
    const std::string& filename, size_t* num_rows, size_t* num_cols,
    std::vector<triplet_d>* entries_double) {
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
    entries_double->emplace_back(rows[iter], cols[iter], vals[iter]);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

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

//   std::vector<size_t> float_times_ns;
  std::vector<size_t> double_times_ns;
  std::vector<double> residual_norms;
  std::random_shuffle(matrix_files.begin(), matrix_files.end());
  for (const std::string& file : matrix_files) {
    size_t num_cols, num_rows;
    std::vector<triplet_d> triplets_d;

    ReadSparseMatrix(file, &num_rows, &num_cols, &triplets_d);
    sparse_qr::SparseSystemDouble system_d(num_rows, num_cols, triplets_d);

    system_d.TimeSolveN(kNumSolves, &double_times_ns, &residual_norms);
    LOG(INFO) << "Finished solves: " << file;
  }
  CHECK_GT(double_times_ns.size(), 0);
  CHECK_EQ(double_times_ns.size(), residual_norms.size());

  // Save everything.
  std::ofstream outfile(time_file);
  for (const auto& time : double_times_ns) {
    outfile << time << std::endl;
  }
  outfile.close();

  outfile.open(res_file);
  for (const auto& res : residual_norms) {
    outfile << std::setprecision(15) << res << std::endl;
  }
}
