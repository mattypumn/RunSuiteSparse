
#include <exception>
#include <fstream>
#include <iomanip>

#include <Eigen/Core>
#include <glog/logging.h>

#include "../include/eigen_helpers/eigen_io.h"
#include "../include/eigen_helpers/stl_io.h"

namespace eigen_helpers {

bool dlmread(const std::string& file_name, Eigen::MatrixXd* matrix) {
  std::ifstream file;
  file.open(file_name);

  if (!file.is_open()) {
    return false;
  }

  std::vector<std::vector<double>> rows;

  while (!file.eof()) {
    std::string line;
    std::getline(file, line);
    size_t cols = 0;
    std::stringstream stream(line);

    rows.push_back(std::vector<double>());
    while (!stream.eof()) {
      double value;

      if(stream.peek() == ',') {
        stream.ignore();
      }

      if (!(stream >> value)) {
        break;
      }
      rows.back().push_back(value);
      cols++;
    }

    if (!cols) {
      rows.pop_back();
    }

    if( rows.size() > 1 ) {
      // Check we have the same number of columns
      if (rows[rows.size()-2].size() != rows[rows.size()-1].size()) {
        return false;
      }
    }
  }

  size_t num_rows = rows.size();
  size_t num_cols = rows[0].size();

  matrix->resize(num_rows, num_cols);

  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_cols; j++) {
      (*matrix)(i, j) = rows[i][j];
    }
  }

  return true;
}

void dlmwrite(
    const std::string& outfile, const Eigen::MatrixXd& mat,
    const std::string& delim) {
  std::ofstream os(outfile.c_str());
  if (!os.is_open()) {
    LOG(FATAL) << "Could not write to file: " << outfile;
  }
  const int cols = mat.cols(), rows = mat.rows();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      os << std::setprecision(15) << mat(i,j) <<
            ((j + 1 == cols) ? "\n" : delim);
    }
  }
}

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

  if (!file.is_open()) {
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
