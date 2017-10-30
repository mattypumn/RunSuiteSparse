
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <glog/logging.h>
#include <gflags/gflags.h>


#include "../../eigen_helpers/include/eigen_helpers/eigen_io.h"
#include "../include/sparse_qr/sparse_system_double.h"


constexpr size_t kNumSolves = 100;
constexpr size_t kJacobianSplitDivisor = 2;
constexpr bool kLoadTranspose = true;
constexpr double kNanoToSeconds = 1e-9;

const std::string time_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/double_times.txt";
const std::string res_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/double_res.txt";



void EigenSparseToTriplets(
    const Eigen::SparseMatrix<double>& S,
    std::vector<sparse_qr::SparseSystemDouble::Triplet>* triplets) {
  const size_t nnz = S.nonZeros();
  triplets->clear();
  triplets->reserve(nnz);
  for (auto k = 0; k < S.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(S,k); it; ++it) {
      triplets->emplace_back(it.row(), it.col(), it.value());
    }
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
  LOG(INFO) << "Input matrix files count: " << matrix_files.size();

  std::vector<size_t> double_times_ns;
  std::vector<double> residual_norms;


  for (const std::string& filename : matrix_files) {
    // Read the matrix.
    const Eigen::SparseMatrix<double> J_t =
        eigen_helpers::ReadSparseMatrix(filename);
    if (kLoadTranspose) {
      LOG(WARNING) << "Loading the transpose of a matrix.";
    }
    const Eigen::SparseMatrix<double> J = kLoadTranspose ?
                                          J_t.transpose() : J_t;
    LOG(INFO) << "Matrix read: " << filename;
    LOG(INFO) << "Matrix size: " << J.rows() << " " << J.cols();

    // Split the matrix into multiple sub-matrices.
    const size_t total_rows = J.rows();
    const size_t total_cols = J.cols();
    const size_t small_rows = total_rows / kJacobianSplitDivisor;
    const size_t last_rows = total_rows -
                            (kJacobianSplitDivisor - 1) * small_rows;
    const size_t num_matrix_blocks = kJacobianSplitDivisor;
    std::vector<Eigen::SparseMatrix<double>> split_matrices;

    CHECK_GT(small_rows, total_cols) << "Split matrices are not full rank"
        "--size: " << small_rows << " " << total_cols <<
        " Try reducing kJacobianSplitDivisor (" << kJacobianSplitDivisor << ")";
    // TODO(mpoulter) Combine last two matrices if final matrix becomes LID.
    CHECK_GT(last_rows, total_cols) << "Final matrix dimension error. size: "
                << last_rows << " " << total_cols;

    for (size_t mat_i = 0; mat_i < num_matrix_blocks - 1; ++ mat_i) {
      const size_t block_start = mat_i * small_rows;
      split_matrices.push_back(J.middleRows(block_start, small_rows));
    }
    split_matrices.push_back(J.middleRows((num_matrix_blocks-1) * small_rows,
                                          last_rows));

    // Assert that the last rows are the identical.
    CHECK_EQ((split_matrices.back().bottomRows(1) -
              J.bottomRows(1)).squaredNorm(), 0);

    // Time giant matrix.
    std::vector<sparse_qr::SparseSystemDouble::Triplet> triplets;
    EigenSparseToTriplets(J, &triplets);
    sparse_qr::SparseSystemDouble* J_solver =
        new sparse_qr::SparseSystemDouble(J.rows(), J.cols(), triplets);
    double J_residual_norm = 0.;
    LOG(INFO) << "Solving Original system...";
    const size_t J_time_ns = J_solver->TimeSolve(&J_residual_norm);
    LOG(INFO) << "J QR-solve time (s): " << J_time_ns * kNanoToSeconds;
    delete J_solver;
    J_solver = nullptr;

    // Time sub-matrices.
    LOG(INFO) << "Solving sub-systems...";
    std::vector<double> split_residual_norms;
    std::vector<double> split_times_ns;
    for (const auto& mat : split_matrices) {
      triplets.clear();
      EigenSparseToTriplets(mat, &triplets);
      sparse_qr::SparseSystemDouble mat_solver(mat.rows(), mat.cols(),
                                               triplets);
      split_residual_norms.push_back(0);
      const size_t time_ns = mat_solver.TimeSolve(
                                  &(split_residual_norms.back()));
      LOG(INFO) << "sub-matrix time (s): " << time_ns * kNanoToSeconds;
      split_times_ns.push_back(time_ns);
    }

//     // Compare norms.
//     const double Jnorm_2 = J_residual_norm * J_residual_norm;
//     double split_sq_sum = 0;
//     for (const auto& res : split_residual_norms) {
//       split_sq_sum += res * res;
//     }
//     LOG(INFO) << "Residual squaredNorm from solving J: " << Jnorm_2;
//     LOG(INFO) << "Sum residual squaredNorm over split matrices: " <<
//                   split_sq_sum;
  }
}
