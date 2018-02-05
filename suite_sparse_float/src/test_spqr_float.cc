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
#include "../include/sparse_qr/sparse_system_float.h"
#include "../../fs_utils/include/fs_utils/fs_utils.h"

constexpr size_t kNumSolves = 5;
constexpr bool kLoadTranspose = true;
constexpr bool kDoThinQR = true;
constexpr bool kAllowPermutations = false;
constexpr double kNanoToSeconds = 1e-9;

typedef sparse_qr::SparseSystemFloat::Triplet SparseSystemTriplet;

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
  LOG(INFO) << "J size: (" << J.rows() << " x " << J.cols() << ")";

  Eigen::MatrixXd B_double;
  Eigen::MatrixXf B_float;

  const bool successful_read = eigen_helpers::dlmread(rhs_filepath, &B_double);
  if (!successful_read) {
    LOG(FATAL) << "Could not read RHS matrix file: " << rhs_filepath;
  }
  LOG(INFO) << "Loaded rhs: " << rhs_filepath;
  B_float.resize(B_double.rows(), B_double.cols());
  for (int i = 0; i < B_double.rows(); ++i) {
    for (int j = 0; j < B_double.cols(); ++j) {
      B_float(i,j) = static_cast<float>(B_double(i, j));
    }
  }

  if (!fsutils::CheckDirectoryExists(output_directory)) {
    LOG(FATAL) << "Cannot locate output directory: " << output_directory;
  }
  LOG(INFO) << "Output directory found: " << output_directory;

  /////////////////////////////////////////////
  //       Solve with SuiteSparse.           //
  /////////////////////////////////////////////
  std::vector<SparseSystemTriplet> mat_triplets;
  std::vector<float> Qt_b_vec;
  std::vector<SparseSystemTriplet> R_trip;
  std::vector<size_t> perm;
  std::vector<double> times_s;
  Eigen::MatrixXd Qt_B_matrix;
  if (kDoThinQR) {
    Qt_B_matrix.resize(J.cols(), B_float.cols());
  } else {
    Qt_B_matrix.resize(B_float.rows(), B_float.cols());
  }
  Eigen::VectorXd p_vec(J.cols());
  EigenSparseToTriplets(J, &mat_triplets);
  LOG(INFO) << "Starting solves.";
  for (size_t test_i = 0; test_i < kNumSolves; ++test_i) {
    for (int system_i = 0; system_i < B_float.cols(); ++system_i) {
      Qt_b_vec.clear();
      R_trip.clear();
      perm.clear();
      Eigen::VectorXf b_col = B_float.col(system_i);
      std::vector<float> b_vec;
      for (int b_elem_i = 0; b_elem_i < b_col.rows(); ++b_elem_i) {
         b_vec.emplace_back(b_col(b_elem_i));
      }
      sparse_qr::SparseSystemFloat* system_solver =
            new sparse_qr::SparseSystemFloat(J.rows(), J.cols(), mat_triplets);
      system_solver->SetPermutations(false);
      system_solver->SetEconomic(true);
      system_solver->SetRhs(b_vec);
      const size_t J_time_ns = system_solver->TimeQrDecomposition(&Qt_b_vec,
                                                      &R_trip, &perm);
      for (int qtb_i = 0; qtb_i < Qt_b_vec.size(); ++qtb_i) {
        Qt_B_matrix(qtb_i, system_i) = Qt_b_vec[qtb_i];
      }
      for (int p_idx = 0; p_idx < perm.size(); ++p_idx) {
        p_vec(p_idx) = perm[p_idx];
      }
      times_s.emplace_back(J_time_ns * kNanoToSeconds);
      delete system_solver;
      system_solver = nullptr;
    }
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
  LOG(INFO) << "Saved R matrix.";
  LOG(INFO) << "Q'* B size: (" << Qt_B_matrix.rows() << " x " <<
              Qt_B_matrix.cols() << ")";
  eigen_helpers::dlmwrite(outfile_QtB, Qt_B_matrix);
  LOG(INFO) << "Saved Qt_b matrix.";
  eigen_helpers::dlmwrite(outfile_perm, p_vec);
  LOG(INFO) << "Saved Permutation indices.";

  return 0;
}
