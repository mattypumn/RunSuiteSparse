#ifndef MARS_FSUTILS_H
#define MARS_FSUTILS_H

#include <vector>
#include <string>

#include <Eigen/Core>

/** @namespace fsutils
 * @brief The fsutils namespace provides functionalities that ease the use of file system.
 */

//TODO: Complete documentation of these functions
namespace fsutils
{

int ListEntriesInDirectory(const std::string& dir,
                           std::vector<std::string>& files);

int ListFilesInDirectory(const std::string& dir,
                         std::vector<std::string>& files);

int ListFilesInDirectoryWithExtention(const std::string& dir,
                                      const std::string& file_ext,
                                      std::vector<std::string>& files);

bool CheckFileExists(const std::string& name);

bool CheckDirectoryExists(const std::string& name);

std::string ApplyFileNameToDirectory(const std::string& dir,
                                     const std::string& filename);

std::string GetCurrentDirectory();

bool CreateDirectory(const std::string& dir_name, int mode = 0777);

bool DeleteFile(const std::string& filename);

bool dlmread(const std::string& file_name, Eigen::MatrixXd& matrix);

std::string GetHomeDirectory();

std::string GetFileNameFromPath(const std::string& path,
                                std::string* directory = nullptr);

std::string RemoveExtensionFromFileName(const std::string& filename,
                                        std::string* extension = nullptr);

}; //namespace fsutils

#endif //MARS_FSUTILS_H


