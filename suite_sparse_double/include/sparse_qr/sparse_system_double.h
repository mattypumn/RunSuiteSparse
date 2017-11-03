#ifndef MARS_SPARSE_SYSTEM_DOUBLE
#define MARS_SPARSE_SYSTEM_DOUBLE



#include <tuple>
#include <vector>

#include <SuiteSparseQR.hpp>


namespace sparse_qr {
class SparseSystemDouble{
 public:
  typedef std::tuple<size_t, size_t, double> Triplet;

  SparseSystemDouble(const size_t& rows, const size_t& cols,
                     const std::vector<Triplet>& A_vals,
                     const size_t& num_threads = 0,
                     const size_t& num_cores = 1);

  ~SparseSystemDouble();

  void SetDimensions(const size_t& rows, const size_t& cols);

  void SetWithTuples(const std::vector<Triplet>& vals);

  void SetRhs(const std::vector<double>& rhs);

  size_t TimeSolve(double* residual_norm = nullptr,
                   std::vector<double>* x_solve = nullptr);

  void TimeSolveN(const size_t& n_solves,
                  std::vector<size_t>* times_ns,
                  std::vector<double>* residual_norms = nullptr);

  size_t SystemSolve(const std::vector<double>& rhs, std::vector<double>* x);

  size_t TimeQrDecomposition(std::vector<double>* QT_b,
                             std::vector<Triplet>* R_triplets,
                             std::vector<size_t>* permutation);
 private:
  SparseSystemDouble();

  void CholmodSparseToTriplet(cholmod_sparse* A, std::vector<Triplet>* triplet);

  cholmod_sparse* A_;
  cholmod_dense* b_;
  cholmod_common cc_;

  const size_t num_threads_;
  const size_t num_cores_;
};

}  // namespace sparse_qr.

#endif  // MARS_SPARSE_SYSTEM_DOUBLE



