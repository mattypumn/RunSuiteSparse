#include <fstream>
#include <iomanip>
#include <math.h>
#include <utility>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <glog/logging.h>


#include "../../eigen_helpers/include/eigen_helpers/eigen_io.h"
#include "../include/sparse_qr/sparse_system_float.h"
#include "../../fs_utils/include/fs_utils/fs_utils.h"

namespace android_suite_sparse {

constexpr size_t kNumSolves = 5;
constexpr bool kDoThinQR = true;
constexpr bool kAllowPermutations = false;
constexpr double kNanoToSeconds = 1e-9;

typedef sparse_qr::SparseSystemFloat::Triplet SparseSystemTriplet;

std::pair<double, double> GetTimeStats(std::vector<double>& times_seconds) {
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
  return std::make_pair(average, std_dev);
}

void EigenSparseToTriplets(
    const Eigen::SparseMatrix<double>& S,
    std::vector<SparseSystemTriplet>* triplets) {
  const size_t nnz = S.nonZeros();
  triplets->clear();
  triplets->reserve(nnz);
  for (auto k = 0; k < S.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(S,k); it; ++it) {
      triplets->emplace_back(it.row(), it.col(),
                             static_cast<float>(it.value()));
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
    const double val = static_cast<double>(std::get<2>(triplet));
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
    LOG(ERROR) << "Usage: <" << prog <<
      "> <bool-do-load-binary-matrix-transpose> "
      "<input-binary-matrix-file> "
      "<input-ascii-matrix-rhs> "
      "<output-directory>" <<
      std::endl << std::endl <<
      "  This program is to be used in conjunction with it's float equivalent "
      "as well as the $$MATLAB_SCRIPT$$ to compare the results."<<
      std::endl << std::endl;
}

std::pair<double, double> TimeQR(const std::string& sparse_filepath) {
  std::string directory;
  const std::string name_and_ext = fsutils::GetFileNameFromPath(sparse_filepath,
                                                            &directory);
  const std::string mat_name = fsutils::RemoveExtensionFromFileName(
                                                        name_and_ext);
  const std::string rhs_file = directory + "/" + mat_name + "_random_rhs.txt";
  const std::string R_file = directory + "/" + mat_name + "_R_float.dat";
  const std::string QtB_file = directory + "/" + mat_name + "_QtB_float.txt";
  const std::string perm_file = directory + "/" + mat_name + "_perm_float.txt";
  LOG(INFO) << "directory: " << directory;
  LOG(INFO) << "file: " << name_and_ext;
  LOG(INFO) << "mat_name: " << mat_name;
  LOG(INFO) << "R file: " << R_file;
  LOG(INFO) << "QtB file: " << QtB_file;
  LOG(INFO) << "perm_file: " << perm_file;

  /////////////////////////////////////////////
  //       Load the Data.                    //
  /////////////////////////////////////////////
  const Eigen::SparseMatrix<double> J =
                eigen_helpers::ReadSparseMatrix(sparse_filepath);
  Eigen::MatrixXd B_double;
  Eigen::MatrixXf B_float;

  const bool successful_read = eigen_helpers::dlmread(rhs_file, &B_double);
  if (!successful_read) {
    LOG(FATAL) << "Could not read RHS matrix file: " << rhs_file;
  }
  LOG(INFO) << "Loaded rhs: " << rhs_file;
  B_float.resize(B_double.rows(), B_double.cols());
  for (int i = 0; i < B_double.rows(); ++i) {
    for (int j = 0; j < B_double.cols(); ++j) {
      B_float(i,j) = static_cast<float>(B_double(i, j));
    }
  }

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
  const std::pair<double, double> stats = GetTimeStats(times_s);

  /////////////////////////////////////////////
  //       Save Data.                        //
  /////////////////////////////////////////////
  Eigen::SparseMatrix<double> R_sparse;
  TripletsToEigenSparse(R_trip, J.cols(), J.cols(), &R_sparse);
  eigen_helpers::WriteSparseMatrix(R_file, R_sparse);
  LOG(INFO) << "Saved R matrix.";
  LOG(INFO) << "Q'* B size: (" << Qt_B_matrix.rows() << " x " <<
              Qt_B_matrix.cols() << ")";
  eigen_helpers::dlmwrite(QtB_file, Qt_B_matrix);
  LOG(INFO) << "Saved Qt_b matrix.";
  eigen_helpers::dlmwrite(perm_file, p_vec);
  LOG(INFO) << "Saved Permutation indices.";

  return stats;
}

}  // namespace android_suite_sparses