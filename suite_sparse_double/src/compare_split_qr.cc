#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <glog/logging.h>
#include <gflags/gflags.h>


#include "../../eigen_helpers/include/eigen_helpers/eigen_io.h"
#include "../include/sparse_qr/sparse_system_double.h"


constexpr size_t kNumThreads = 1;
constexpr size_t kNumCores = 1;
constexpr size_t kNumSolves = 100;
constexpr size_t kJacobianSplitDivisor = 2;
constexpr bool kLoadTranspose = true;
constexpr double kNanoToSeconds = 1e-9;

const std::string time_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/double_times.txt";
const std::string res_file = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/double_res.txt";
const std::string out_dir = "/usr/local/google/home/mpoulter/RunSuiteSparse/data/";
const std::string R_file = "R";
const std::string QT_b_file = "QT_b";
const std::string Perm_file = "Permutation";
const std::string x_file = "x_solved";
const std::string out_ext = ".mat";

typedef sparse_qr::SparseSystemDouble::Triplet SparseSystemTriplet;

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

void WriteDecomposition(
    const size_t&  id, const std::vector<double>& QT_b,
    const size_t& R_rows, const size_t& R_cols,
    const std::vector<SparseSystemTriplet>& R_trips,
    const std::vector<size_t>& perm) {
  LOG(WARNING) << "Saving decomposition to file.";
  std::stringstream ss;
  ss << out_dir << R_file << "_" << id << out_ext;
  const std::string R_filename = ss.str();
  ss.str(std::string());
  ss << out_dir << QT_b_file << "_" << id << out_ext;
  const std::string QTB_filename = ss.str();
  ss.str(std::string());
  ss << out_dir << Perm_file << "_" << id << out_ext;
  const std::string Perm_filename = ss.str();

  Eigen::SparseMatrix<double> R_sparse;
  TripletsToEigenSparse(R_trips, R_rows, R_cols, &R_sparse);
  eigen_helpers::WriteSparseMatrix(R_filename, R_sparse);
  LOG(INFO) << "Finished writing " << R_filename;
  std::ofstream write_fid(QTB_filename);
  for (const auto& val : QT_b) {
    write_fid << std::setprecision(15) << val << std::endl;
  }
  write_fid.close();
  LOG(INFO) << "Finished writing " << QTB_filename;
  write_fid.open(Perm_filename);
  for (const auto& val : perm) {
    write_fid << val << std::endl;
  }
  write_fid.close();
  LOG(INFO) << "Finished writing " << Perm_filename;
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  if (argc < 2) {
    LOG(ERROR) << "Usage: <" << argv[0] << "> <binary-matrix-file-1> ...";
    return -1;
  }

  LOG(INFO) << "Number of cores to use: " << kNumCores;
  LOG(INFO) << "Number of threades to use: " << kNumThreads;

  std::vector<std::string> matrix_files;
  for (int i = 1; i < argc; ++i) {
    matrix_files.push_back(argv[i]);
  }
  LOG(INFO) << "Input matrix files count: " << matrix_files.size();


  /////////////////////////////////////////////
  //           Loop over Jacobians.          //
  /////////////////////////////////////////////
  std::vector<size_t> double_times_ns;
  std::vector<double> residual_norms;
  for (const std::string& filename : matrix_files) {
    /////////////////////////////////////////////
    //           Load the matrix.              //
    /////////////////////////////////////////////
    const Eigen::SparseMatrix<double> J_t =
        eigen_helpers::ReadSparseMatrix(filename);
    if (kLoadTranspose) {
      LOG(WARNING) << "Loading the transpose of a matrix.";
    }
    const Eigen::SparseMatrix<double> J = kLoadTranspose ?
                                          J_t.transpose() : J_t;
    LOG(INFO) << "Matrix read: " << filename;
    LOG(INFO) << "Matrix size: " << J.rows() << " " << J.cols();
    LOG(INFO) << "Matrix nnz: " << J.nonZeros();

    /////////////////////////////////////////////
    //           Split J into submatrices.     //
    /////////////////////////////////////////////
    const size_t total_rows = J.rows();
    const size_t total_cols = J.cols();
    const size_t small_rows = total_rows / kJacobianSplitDivisor;
    const size_t last_rows = total_rows -
                            (kJacobianSplitDivisor - 1) * small_rows;
    const size_t num_matrix_blocks = kJacobianSplitDivisor;
    std::vector<Eigen::SparseMatrix<double>> split_matrices;

    CHECK_GT(small_rows, total_cols) << "Split matrices are too small. "
        "--size: " << small_rows << " " << total_cols <<
        " Try reducing kJacobianSplitDivisor (" << kJacobianSplitDivisor << ")";
    // TODO(mpoulter) Combine last two matrices if final matrix becomes to,
    // short.
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
              J.bottomRows(1)).squaredNorm(), 0) << "Error Saving sub-matrices."
              "  The final row of last matrix and J are not the same.";

    /////////////////////////////////////////////
    //           Time J.                       //
    /////////////////////////////////////////////
    std::vector<sparse_qr::SparseSystemDouble::Triplet> mat_triplets;
    EigenSparseToTriplets(J, &mat_triplets);
    sparse_qr::SparseSystemDouble* system_solver =
        new sparse_qr::SparseSystemDouble(J.rows(), J.cols(), mat_triplets,
                                          kNumThreads, kNumCores);
    std::vector<double> QT_b;
    std::vector<SparseSystemTriplet> R_trip;
    std::vector<size_t> perm;
    LOG(INFO) << "Solve for R and Q'*b of the original system.";
    std::vector<double> x_solve;

    // Solve the system.
    // NOTE  This system solve is still working properly.
    //       (Everything to this point is working correctly.)
    // const size_t J_time_ns =  system_solver->TimeSolve(&residual, &x_solve);

    // Solve for the QR.
    const size_t J_time_ns = system_solver->TimeQrDecomposition(&QT_b, &R_trip,
                                                                &perm);
    LOG(INFO) << "Full system time (s): " << J_time_ns  * kNanoToSeconds;
    delete system_solver;
    system_solver = nullptr;

    size_t matrix_counter = 0;

    // Save the decomposition and Q'*b.
//     WriteDecomposition(matrix_counter, QT_b, J.cols(), J.cols(), R_trip,
//                        perm);
//     LOG(FATAL) << "This program is not recovering our R matrix properly." <<
//                   std::endl << "Use matlab script 'spy_R.m' to compare"
//                   "with the matrix solved using spqr in matlab.";
//     ++matrix_counter;

    /////////////////////////////////////////////
    //           Time the submatrices.         //
    /////////////////////////////////////////////
    LOG(INFO) << "Solving sub-systems...";
    const size_t R_block_size = J.cols();
    size_t m_huge = 0;
    for (const auto& mat : split_matrices) {
      m_huge += mat.rows();
    }
    CHECK_LE(m_huge, num_matrix_blocks * R_block_size);
    Eigen::SparseMatrix<double> R_huge(m_huge, J.cols());
    std::vector<SparseSystemTriplet> R_huge_triplets;
    std::vector<double> C_huge(m_huge);  // C = Q'*b
    std::vector<double> split_times_ns;
    for (size_t mat_i = 0; mat_i < num_matrix_blocks; ++mat_i) {
      mat_triplets.clear();
      QT_b.clear();
      R_trip.clear();
      perm.clear();
      const size_t block_start = mat_i * R_block_size;
      const auto& mat = split_matrices[mat_i];
      EigenSparseToTriplets(split_matrices[mat_i], &mat_triplets);
      sparse_qr::SparseSystemDouble mat_solver(mat.rows(), mat.cols(),
                                               mat_triplets);
      const size_t time_ns = mat_solver.TimeQrDecomposition(&QT_b, &R_trip,
                                                            &perm);
      LOG(INFO) << matrix_counter << " sub-matrix time (s): " <<
                   time_ns * kNanoToSeconds;
      LOG(INFO) << "nnz to add to R_huge: " << R_trip.size();

      //  Get the Sub Matrix.
      Eigen::SparseMatrix<double> R_sub;
      TripletsToEigenSparse(R_trip, R_block_size, R_block_size, &R_sub);

      // Apply Permutation.
      Eigen::VectorXi p_vec(perm.size());
      for (int perm_i = 0; perm_i < perm.size(); ++ perm_i) {
        p_vec(perm_i) = static_cast<int>(perm[perm_i]);
      }
      R_sub = R_sub * p_vec.asPermutation().inverse();

      // Extract triplets after permutation.
      std::vector<sparse_qr::SparseSystemDouble::Triplet> tmp_trips;
      EigenSparseToTriplets(R_sub, &tmp_trips);

      // Add triplets to R_huge, remembering block offset.
      R_huge_triplets.reserve(tmp_trips.size());
      // TODO(mpoulter) Use Functor to help with parallelization.
      for (const auto& entry : R_trip) {
        const size_t r_row = std::get<0>(entry);
        const size_t r_col = std::get<1>(entry);
        const double value = std::get<2>(entry);
        const size_t giant_row = (R_block_size * mat_i) + r_row;
        const size_t giant_col = r_col;
        CHECK_NE(value, 0.0);
        CHECK_LT(giant_col, R_block_size) << " Column is out of bounds.";
        CHECK_LT(giant_row, m_huge) << " Row is out of bounds.";
        R_huge_triplets.emplace_back(giant_row, giant_col, value);
      }
      LOG(INFO) << "R_huge_triplets size: " << R_huge_triplets.size();

      for (size_t qtb_i = 0; qtb_i < QT_b.size(); ++qtb_i) {
        const size_t C_huge_idx = block_start + qtb_i;
        C_huge[C_huge_idx] = QT_b[qtb_i];
      }

      WriteDecomposition(matrix_counter, QT_b, J.cols(), J.cols(), R_trip,
                         perm);
      ++matrix_counter;
    }  // End for each matrix.

  QT_b.clear();
  R_trip.clear();
  perm.clear();
  system_solver = new sparse_qr::SparseSystemDouble(m_huge, J.cols(),
                            R_huge_triplets, kNumThreads, kNumCores);
  LOG(INFO) << "Solve for R and Q'*b of the Final system.";
  system_solver->SetRhs(C_huge);
  std::vector<double> final_solution;
  double final_residual;
  const size_t R_huge_time_ns = system_solver->TimeSolve(&final_residual,
                                                         &final_solution);

  LOG(INFO) << "J solve time (s): " << R_huge_time_ns * kNanoToSeconds;
  LOG(INFO) << "Final residual: " << final_residual;
  delete system_solver;
  system_solver = nullptr;

  LOG(INFO) << "Saving solution";
  std::ofstream write_fid(out_dir + x_file + out_ext);
  for (const auto& xi : final_solution) {
    write_fid << xi << std::endl;
  }
  write_fid.close();


  ////////////////////////////////////////
  // Calculate final residual.
  ////////////////////////////////////////


  }
}
