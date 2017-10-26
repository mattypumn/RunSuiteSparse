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
                     const std::vector<Triplet>& vals);
  ~SparseSystemDouble();

  void SetDimensions(const size_t& rows, const size_t& cols);
  void SetWithTuples(const std::vector<Triplet>& vals);
  void SetRhs(std::vector<double> rhs);

  size_t TimeSolve();

  void TimeSolveN(const size_t& n_solves, std::vector<size_t>* times_ns);

  size_t SystemSolve(const std::vector<double>& rhs, std::vector<double>* x);
 private:
  SparseSystemDouble();

//   cholmod_dense* sparse_qr(cholmod_sparse* A, cholmod_dense* b,
//                            cholmod_common* cc);

  cholmod_dense* sparse_qr();

  cholmod_sparse* A_;
  cholmod_dense* b_;
  cholmod_common cc_;
};

}  // namespace sparse_qr.

#endif  // MARS_SPARSE_SYSTEM_DOUBLE



