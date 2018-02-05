#ifndef MARS_SPARSE_SYSTEM_FLOAT
#define MARS_SPARSE_SYSTEM_FLOAT

#include <SuiteSparseQR.hpp>
#include <tuple>
#include <vector>

namespace sparse_qr {

class SparseSystemFloat{
 public:
  typedef std::tuple<size_t, size_t, float> Triplet;

  SparseSystemFloat(const size_t& rows, const size_t& cols,
                     const std::vector<Triplet>& vals);
  ~SparseSystemFloat();

  void SetDimensions(const size_t& rows, const size_t& cols);
  void SetWithTuples(const std::vector<Triplet>& vals);
  void SetRhs(std::vector<float> rhs);

  size_t TimeSolve(float* residual = nullptr);

  void TimeSolveN(const size_t& n_solves, std::vector<size_t>* times_ns,
                  std::vector<float>* residual_norms = nullptr);

  size_t SystemSolve(const std::vector<float>& rhs, std::vector<float>* x);

  size_t TimeQrDecomposition(std::vector<float>* QT_b,
                             std::vector<Triplet>* R_triplets,
                             std::vector<size_t>* permutation);

  void SetPermutations(bool do_permutations) {
    do_permutations_ = do_permutations;
  }

  void SetEconomic(bool econ) {
    solve_economy_ = econ;
  }

 private:
  // Purposefully hidden constructor.
  SparseSystemFloat();


  void CholmodSparseToTriplet(cholmod_sparse* A, std::vector<Triplet>* triplet);
  
  cholmod_dense* SolveQR();
  cholmod_sparse* A_;
  cholmod_dense* b_;
  cholmod_common cc_;
  bool do_permutations_;
  bool solve_economy_;
};

}  // namespace sparse_qr.

#endif  // MARS_SPARSE_SYSTEM_FLOAT
