
#include <exception>
#include <fstream>

#include <Eigen/Core>
#include <glog/logging.h>

#include "../include/eigen_helpers/eigen_io.h"
#include "../include/eigen_helpers/stl_io.h"

namespace eigen_helpers {

void WriteAscii(std::ostream& os, const Eigen::MatrixXd& mat)
{
  os << mat.rows() << " " << mat.cols() << "\n";
  os << mat << "\n";
}

void WriteBinary(std::ostream& os, const Eigen::MatrixXd& mat)
{
  std::uint64_t rows = mat.rows(), cols = mat.cols();
  os.write(reinterpret_cast<char*>(&rows), sizeof(rows));
  os.write(reinterpret_cast<char*>(&cols), sizeof(cols));
  for(Eigen::MatrixXd::Index r = 0; r < rows; r++)
    for(Eigen::MatrixXd::Index c = 0; c < cols; c++)
    {
      Eigen::MatrixXd::Scalar value = mat(r, c);
      os.write(reinterpret_cast<char*>(&value), sizeof(value));
    }
}

void ReadAscii(std::istream& is, Eigen::MatrixXd& mat) {
  std::uint64_t rows, cols;
  is >> rows >> cols;
  mat.resize(rows, cols);
  for(Eigen::MatrixXd::Index r = 0; r < rows; r++)
    for(Eigen::MatrixXd::Index c = 0; c < cols; c++)
      is >> mat(r, c);
}

void ReadBinary(std::istream& is, Eigen::MatrixXd& mat) {
  std::uint64_t rows = mat.rows(), cols = mat.cols();
  is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  mat.resize(rows, cols);
  for(Eigen::MatrixXd::Index r = 0; r < rows; r++)
    for(Eigen::MatrixXd::Index c = 0; c < cols; c++)
    {
      Eigen::MatrixXd::Scalar value;;
      is.read(reinterpret_cast<char*>(&value), sizeof(value));
      mat(r, c) = value;
    }
}

Eigen::MatrixXd ReadMatrixAscii(const std::string& filename)
{
  std::ifstream file(filename);
  if( !file )
    throw std::runtime_error("(ReadMatrixAscii) Unable to open file: " + filename);
  Eigen::MatrixXd mat;
  ReadAscii(file, mat);
  return mat;
}

Eigen::MatrixXd ReadMatrixBinary(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if( !file )
    throw std::runtime_error("(ReadMatrixBinary) Unable to open file: " + filename);
  Eigen::MatrixXd mat;
  ReadBinary(file, mat);
  return mat;
}

void WriteMatrixAscii(const std::string& filename, const Eigen::MatrixXd& mat)
{
  std::ofstream file(filename);
  if( !file )
    throw std::runtime_error("(WriteMatrixAscii) Unable to open file: " + filename);
  WriteAscii(file, mat);
}

void WriteMatrixBinary(const std::string& filename, const Eigen::MatrixXd& mat)
{
  std::ofstream file(filename);
  if( !file )
    throw std::runtime_error("(WriteMatrixBinary) Unable to open file: " + filename);
  WriteBinary(file, mat);
}

void WriteSparseMatrix(const std::string& filename, const Eigen::SparseMatrix<double>& mat)
{
  std::ofstream file(filename);
  if( !file )
    throw std::runtime_error("(WriteSparseMatrix) Unable to open file: " + filename);

  std::uint64_t nnz = mat.nonZeros();
  std::vector<std::uint64_t> rows(nnz), cols(nnz);
  std::vector<double> vals(nnz);

  size_t index = 0;
  for(auto k = 0; k < mat.outerSize(); k++)
    for(Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it)
    {
      rows[index] = it.row();
      cols[index] = it.col();
      vals[index] = it.value();
      index++;
    }

  std::uint64_t num_rows, num_cols;
  num_rows = mat.rows();
  num_cols = mat.cols();

  file.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
  file.write(reinterpret_cast<const char*>(&num_cols), sizeof(num_cols));
  file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));

  WriteBinary(file, rows);
  WriteBinary(file, cols);
  WriteBinary(file, vals);
}

Eigen::SparseMatrix<double> ReadSparseMatrix(const std::string& filename) {
  std::ifstream file(filename);
  std::vector<std::uint64_t> rows, cols;
  std::vector<double> vals;
  std::uint64_t num_rows, num_cols, nnz;

  if (file == nullptr) {
    throw std::runtime_error("(ReadSparseMatrix) Unable to read file: " + filename);
  }

  file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
  file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
  file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
  ReadBinary(file, rows);
  ReadBinary(file, cols);
  ReadBinary(file, vals);

  CHECK_EQ(rows.size(), cols.size());
  CHECK_EQ(rows.size(), vals.size());

  std::vector<Eigen::Triplet<double>> triplets;
  for (size_t iter = 0; iter < rows.size(); ++iter) {
    CHECK_LT(rows[iter], num_rows);
    CHECK_LT(cols[iter], num_cols);
    triplets.emplace_back(rows[iter], cols[iter], vals[iter]);
  }

  Eigen::SparseMatrix<double> mat(num_rows, num_cols);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}

}; // end namespace eigen_helpers
