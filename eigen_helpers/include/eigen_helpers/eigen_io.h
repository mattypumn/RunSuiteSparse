
#ifndef BLS_IO_EIGEN_IO_H_
#define BLS_IO_EIGEN_IO_H_

#include <fstream>
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace eigen_helpers {

bool dlmread(const std::string& file_name, Eigen::MatrixXd* matrix);
void dlmwrite(const std::string& outfile, const Eigen::MatrixXd& mat,
              const std::string& delim = ",");

void WriteAscii(std::ostream& os, const Eigen::MatrixXd& mat);
void WriteBinary(std::ostream& os, const Eigen::MatrixXd& mat);

void ReadAscii(std::istream& is, Eigen::MatrixXd& mat);
void ReadBinary(std::istream& is, Eigen::MatrixXd& mat);

Eigen::MatrixXd ReadMatrixAscii(const std::string& filename);
Eigen::MatrixXd ReadMatrixBinary(const std::string& filename);

void WriteMatrixAscii(const std::string& filename, const Eigen::MatrixXd& mat);
void WriteMatrixBinary(const std::string& filename, const Eigen::MatrixXd& mat);

void WriteSparseMatrix(const std::string& filename, const Eigen::SparseMatrix<double>& mat);
Eigen::SparseMatrix<double> ReadSparseMatrix(const std::string& filename);

}; // end namespace eigen_helpers

#endif //BLS_IO_EIGEN_IO_H_
