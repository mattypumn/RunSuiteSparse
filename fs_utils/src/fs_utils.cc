#include "../include/fs_utils/fs_utils.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>
#include <fstream>

namespace fsutils
{

int ListEntriesInDirectory(const std::string& dir, std::vector<std::string>& files) {
  files.clear();
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL)
    return errno;

  while( (dirp = readdir(dp)) != NULL )
    files.push_back(dirp->d_name);

  closedir(dp);
  return 0;
}

int ListFilesInDirectory(const std::string& dir, std::vector<std::string>& files) {
  files.clear();
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == nullptr) {
    return errno;
  }

  while((dirp = readdir(dp)) != nullptr) {
    if (dirp->d_type == DT_REG) {
      files.push_back(dirp->d_name);
    }
  }

  closedir(dp);
  return 0;
}

static bool EndsWith(const std::string& str, const std::string& ending)
{
  size_t str_len = str.length();
  size_t ending_len = ending.length();

  if( str_len >= ending_len )
    return (str.compare(str_len - ending_len, ending_len, ending) == 0);
  else
    return false;
}

int ListFilesInDirectoryWithExtention(const std::string& dir, const std::string& file_ext, std::vector<std::string>& files)
{
  std::vector<std::string> all_files;
  int err_val;

  if( (err_val = ListFilesInDirectory(dir, all_files)) != 0 )
    return err_val;

  files.reserve(all_files.size());
  for(const auto& filename: all_files)
    if( EndsWith(filename, file_ext) )
      files.push_back(filename);
  return 0;
}

bool CheckFileExists(const std::string& name)
{
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

bool CheckDirectoryExists(const std::string& name)
{
  struct stat buffer;
  return ((stat(name.c_str(), &buffer) == 0) && S_ISDIR(buffer.st_mode));
}

std::string ApplyFileNameToDirectory(const std::string& dir, const std::string& filename)
{
  if( EndsWith(dir, "/") || (filename.length() > 0 && filename[0] == '/') )
  {
    return dir + filename;
  }
  else
  {
    return dir + "/" + filename;
  }
}

#define MAXPATHLEN  1024
std::string GetCurrentDirectory()
{
  char temp[MAXPATHLEN];
  return (getcwd(temp, MAXPATHLEN) ? std::string(temp) : std::string(""));
}

bool DeleteFile(const std::string& filename)
{
  return (remove(filename.c_str()) == 0);
}


bool dlmread(const std::string& file_name, Eigen::MatrixXd& matrix)
{
  std::ifstream file;
  file.open(file_name);

  if( !file )
    return false;

  std::vector<std::vector<double>> rows;

  while( !file.eof() )
  {
    std::string line;
    std::getline(file, line);
    size_t cols = 0;
    std::stringstream stream(line);

    rows.push_back(std::vector<double>());
    while( !stream.eof() )
    {
      double value;

      if( stream.peek() == ',' )
        stream.ignore();

      if( !(stream >> value) )
        break;
      rows.back().push_back(value);
      cols++;
    }

    if( !cols )
      rows.pop_back();

    if( rows.size() > 1 )
    {
      // Check we have the same number of columns
      if( rows[rows.size()-2].size() != rows[rows.size()-1].size() )
        return false;
    }
  }

  size_t num_rows = rows.size();
  size_t num_cols = rows[0].size();

  matrix.resize(num_rows, num_cols);

  for(size_t i = 0; i < num_rows; i++)
    for(size_t j = 0; j < num_cols; j++)
      matrix(i, j) = rows[i][j];

  return true;
}

// The following two functions are taken from: https://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux

static int do_mkdir(const char* path, mode_t mode)
{
  struct stat st;
  int status = 0;

  if (stat(path, &st) != 0)
  {
    /* Directory does not exist. EEXIST for race condition */
    if( mkdir(path, mode) != 0 && errno != EEXIST )
      status = -1;
  }
  else if( !S_ISDIR(st.st_mode) )
  {
    errno = ENOTDIR;
    status = -1;
  }

  return status;
}

static int mkpath(const std::string& path, mode_t mode)
{
  char* pp;
  char* sp;
  int status;
  size_t len = path.length();
  char* copypath = new char[len + 1];
  memcpy(static_cast<void*>(copypath), static_cast<const void*>(path.c_str()), len);
  copypath[len] = '\0';

  status = 0;
  pp = copypath;
  while( status == 0 && (sp = strchr(pp, '/')) != 0 )
  {
    if( sp != pp )
    {
      /* Neither root nor double slash in path */
      *sp = '\0';
      status = do_mkdir(copypath, mode);
      *sp = '/';
    }
    pp = sp + 1;
  }
  if( status == 0 )
    status = do_mkdir(path.c_str(), mode);

  delete[] copypath;
  return status;
}

bool CreateDirectory(const std::string& dir_name, int mode)
{
  return (mkpath(dir_name, mode) == 0);
}

std::string GetHomeDirectory()
{
#ifndef ANDROID
  //FIXME: USE getpwuid_r, since it supports multithreaded applications.
  struct passwd *pw = getpwuid(getuid());
  return pw->pw_dir;
#else
  // TODO: Is there a correct way to do this?

  return "/sdcard/";
#endif
}

std::string GetFileNameFromPath(const std::string& path, std::string* directory) {
  const size_t slash_position = path.find_last_of('/');
  std::string file_name, file_directory;
  if (slash_position == std::string::npos) {
    // no slashes, so the whole thing is the file name.
    file_name = path;
  } else {
    file_name = path.substr(slash_position + 1);
    file_directory = path.substr(0, slash_position);
  }

  if (directory) {
    *directory = file_directory;
  }
  return file_name;
}

std::string RemoveExtensionFromFileName(const std::string& filename, std::string* extension_string) {
  const size_t dot_position = filename.find_last_of('.');
  std::string file_name, extension;
  if (dot_position == std::string::npos) {
    // no slashes, so the whole thing is the file name.
    file_name = filename;
  } else {
    file_name = filename.substr(0, dot_position);
    extension = filename.substr(dot_position + 1);
  }

  if (extension_string) {
    *extension_string = extension;
  }
  return file_name;
}

} //namespace fsutils

