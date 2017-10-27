#ifndef BLS_IO_STL_IO_H_
#define BLS_IO_STL_IO_H_

#include <vector>
#include <sstream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

namespace eigen_helpers {
//NOTE: T should not be a pointer type.
//TODO: Add code to handle if T was pointer or not a base tyoe.
template<typename T>
void WriteAscii(std::ostream& os, const std::vector<T>& v)
{
  os << v.size() << " ";
  for(const T& x : v)
    os << x << " ";
}

template<typename T>
void WriteBinary(std::ostream& os, const std::vector<T>& v)
{
  std::uint64_t sz = v.size();
  os.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  os.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(T));
}

template<typename T>
void ReadAscii(std::istream& is, std::vector<T>& v)
{
  typename std::remove_reference<decltype(v)>::type::size_type sz, i;
  is >> sz;
  v.resize(sz);
  for(i = 0; i < sz; ++i)
    is >> v[i];
}

template<typename T>
void ReadBinary(std::istream& is, std::vector<T>& v)
{
  std::uint64_t sz;
  is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  v.resize(sz);
  is.read(reinterpret_cast<char*>(v.data()), sizeof(T) * sz);
}




template<typename T>
void WriteAscii(std::ostream& os, const std::vector<std::vector<T>>& v)
{
  os << v.size() << " ";
  for(const auto& x : v)
    WriteAscii(os, x);
}

template<typename T>
void WriteBinary(std::ostream& os, const std::vector<std::vector<T>>& v)
{
  std::uint64_t sz = v.size();
  os.write(reinterpret_cast<char*>(&sz), sizeof(sz));
  for(const auto& x : v)
    WriteBinary(os, x);
}

template<typename T>
void ReadAscii(std::istream& is, std::vector<std::vector<T>>& v)
{
  typename std::remove_reference<decltype(v)>::type::size_type sz, i;
  is >> sz;
  v.resize(sz);
  for(i = 0; i < sz; ++i)
    ReadAscii(is, v[i]);
}

template<typename T>
void ReadBinary(std::istream& is, std::vector<std::vector<T>>& v)
{
  std::uint64_t sz, i;
  is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  v.resize(sz);
  for(i = 0; i < sz; ++i)
    ReadBinary(is, v[i]);
}


template<typename T>
std::vector<T> ReadFromFileAscii(const std::string& filename)
{
  std::ifstream file(filename);
  if( !file )
    throw std::runtime_error("(ReadFromFileAscii) Could not open file: " + filename);

  typename std::vector<T> ret;
  std::string line;
  while( std::getline(file, line) )
  {
    T val;
    std::stringstream ss(line);
    ss >> val;
    ret.push_back(val);
  }
  return ret;
}

template<typename T>
std::vector<std::uint64_t> ConvertIndexType(const std::vector<T>& indices)
{
  std::vector<std::uint64_t> retval(indices.size());
  for(size_t i = 0; i < indices.size(); i++)
    retval[i] = static_cast<std::uint64_t>(indices[i]);
  return retval;
}

}; // end namespace eigen_helpers

#endif //BLS_IO_STL_IO_H_

