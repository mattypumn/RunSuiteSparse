#include <fstream>
#include <iomanip>
#include <math.h>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <glog/logging.h>
#include <gflags/gflags.h>


#include "../../eigen_helpers/include/eigen_helpers/eigen_io.h"
#include "../include/sparse_qr/sparse_system_double.h"
#include "../../fs_utils/include/fs_utils/fs_utils.h"

constexpr size_t kNumSolves = 5;
constexpr bool kLoadTranspose = true;
constexpr double kNanoToSeconds = 1e-9;

typedef sparse_qr::SparseSystemDouble::Triplet SparseSystemTriplet;

void ShowTimeStats(std::vector<double>& times_seconds) {
  double total_time = 0.;
  for (const double& time : times_seconds) {
    total_time += time;
  }
  const double average = total_time / times_seconds.size();

  double sqr_diffs = 0;
  for (const double& time : times_seconds) {
    sqr_diffs += (time - average) * (time - average);
  }
  const double std_dev = std::sqrt(sqr_diffs / times_seconds.size());
  LOG(INFO) << "Average solve time: " << average;
  LOG(INFO) << "Standard Deviation: " << std_dev;
}


void EigenSparseToTriplets(
    const Eigen::SparseMatrix<double>& S,
    std::vector<SparseSystemTriplet>* triplets) {
  const size_t nnz = S.nonZeros();
  triplets->clear();
  triplets->reserve(nnz);
  for (auto k = 0; k < S.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(S,k); it; ++it) {
      triplets->emplace_back(it.row(), it.col(), it.value());
    }
  }
}

void TripletsToEigenSparse(
    const std::vector<SparseSystemTriplet>& triplets,
    const size_t& rows, const size_t& cols,
    Eigen::SparseMatrix<double>* S) {
  *S = Eigen::SparseMatrix<double>(rows, cols);
  std::vector<Eigen::Triplet<double>> eigen_triplets;
  eigen_triplets.reserve(triplets.size());
  for (const auto& triplet : triplets) {
    const size_t row = std::get<0>(triplet),
                 col = std::get<1>(triplet);
    const double val = std::get<2>(triplet);
    CHECK_GE(row, 0);
    CHECK_LT(row, rows);
    CHECK_GE(col, 0);
    CHECK_LT(col, cols);
    CHECK_NE(val, 0);
    eigen_triplets.emplace_back(row, col, val);
  }
  S->setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
}

void ShowUsage(char* prog) {
    LOG(ERROR) << "Usage: <" << prog << "> <input-binary-matrix-file> "
      "<input-ascii-matrix-rhs> <output-directory>" <<
      std::endl << std::endl <<
      "  This program is to be used in conjunction with it's float equivalent "
      "as well as the $$MATLAB_SCRIPT$$ to compare the results."<<
      std::endl << std::endl;
}

int main (int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  if (argc != 4) {
    ShowUsage(argv[0]);
    return 0;
  }

  const std::string sparse_filepath(argv[1]);
  const std::string rhs_filepath(argv[2]);
  const std::string output_directory(argv[3]);

  LOG(INFO) << "Sparse matrix file: " << sparse_filepath;
  LOG(INFO) << "rhs: "  << rhs_filepath;
  LOG(INFO) << "Output directory: " << output_directory;

  /////////////////////////////////////////////
  //       Load the Data.                    //
  /////////////////////////////////////////////
  const Eigen::SparseMatrix<double> J_t =
        eigen_helpers::ReadSparseMatrix(sparse_filepath);
  if (kLoadTranspose) {
    LOG(WARNING) << "Loading the transpose of sparse matrix.";
  }
  const Eigen::SparseMatrix<double> J = kLoadTranspose ? J_t.transpose() : J_t;
  LOG(INFO) << "Loaded sparse matrix: " << sparse_filepath;

  Eigen::MatrixXd B;
  fsutils::dlmread(rhs_filepath, B);
  LOG(INFO) << "Loaded rhs: " << rhs_filepath;

  if (!fsutils::CheckDirectoryExists(output_directory)) {
    LOG(FATAL) << "Cannot locate output directory: " << output_directory;
  }

  /////////////////////////////////////////////
  //       Solve with SuiteSparse.           //
  /////////////////////////////////////////////
  std::vector<sparse_qr::SparseSystemDouble::Triplet> mat_triplets;
  std::vector<double> Qt_B;
  std::vector<SparseSystemTriplet> R_trip;
  std::vector<size_t> perm;
  std::vector<double> times_s;
  for (size_t test_i = 0; test_i < kNumSolves; ++test_i) {
    Qt_B.clear();
    R_trip.clear();
    perm.clear();
    EigenSparseToTriplets(J, &mat_triplets);
    sparse_qr::SparseSystemDouble* system_solver =
            new sparse_qr::SparseSystemDouble(J.rows(), J.cols(), mat_triplets);
    const size_t J_time_ns = system_solver->TimeQrDecomposition(&Qt_B, &R_trip,
                                                              &perm);
    times_s.emplace_back(J_time_ns * kNanoToSeconds);
    delete system_solver;
    system_solver = nullptr;
    LOG(INFO) << "solve " << test_i + 1 << " / " << kNumSolves;
  }
  ShowTimeStats(times_s);

  /////////////////////////////////////////////
  //       Save Data.                        //
  /////////////////////////////////////////////
  const std::string name_ext = fsutils::GetFileNameFromPath(sparse_filepath);
  const std::string mat_name = fsutils::RemoveExtensionFromFileName(name_ext);
  const std::string outfile_R = output_directory + "/" + mat_name +
                                "_R_double.dat";
  const std::string outfile_QtB = output_directory + "/" + mat_name +
                                  "_QtB_double.txt";
  const std::string outfile_perm = output_directory + "/" + mat_name +
                                   "_perm_double.txt";

  LOG(INFO) << "Matrix name with extension: " << name_ext;
  LOG(INFO) << "Matrix Name: " << mat_name;
  LOG(INFO) << "outfile_R: " << outfile_R;
  LOG(INFO) << "outfile_QtB: " << outfile_QtB;
  LOG(INFO) << "outfile_perm: " << outfile_perm;

  Eigen::SparseMatrix<double> R_sparse;
  TripletsToEigenSparse(R_trip, J.cols(), J.cols(), &R_sparse);
  eigen_helpers::WriteSparseMatrix(outfile_R, R_sparse);

  std::ofstream write_fid(outfile_QtB);
  for (const auto& val : Qt_B) {
    write_fid << std::setprecision(15) << val << std::endl;
  }


  return 0;
}