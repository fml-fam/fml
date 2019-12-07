#ifndef FML_MEMUSE_H
#define FML_MEMUSE_H


#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>


// -----------------------------------------------------------------------------
// 
// -----------------------------------------------------------------------------

class memuse
{
  public:
    memuse(uint64_t bytes_=0, std::string prefix_="IEC", std::string names_="short");
    memuse& operator=(const memuse& m);
    void set(uint64_t bytes_);
    void swap_prefix();
    void swap_names();
    void print();
    std::string print_str();
    void howbig(uint64_t m, uint64_t n=1, int bytes_per_val=8);
  
  protected:
    uint64_t get_bytes() const {return bytes;};
    std::string get_prefix() const {return prefix;};
    std::string get_names() const {return names;};
  
  private:
    unsigned int bytes;
    double size;
    std::string prefix;
    std::string names;
    int prefix_indx;
    int names_indx;
    int size_indx;
    int denom;
    
    const std::vector<std::vector<std::vector<std::string>>> units_table = {
      {
        {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"},
        {"bytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes", "pebibytes", "exbibytes", "zebibytes", "yobibytes"}
      },
      {
        {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"},
        {"bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "petabytes", "exabytes", "zettabytes", "yottabytes"}
      }
    };
    
    void update_prefix();
    void update_names();
    void update_size();
};



inline memuse::memuse(uint64_t bytes_, std::string prefix_, std::string names_)
{
  if (prefix_ != "IEC" && prefix != "SI")
    throw std::runtime_error("prefix must be \"IEC\" or \"SI\"");
  if (names_ != "short" && names_ != "long")
    throw std::runtime_error("names must be \"short\" or \"long\"");
  
  bytes = bytes_;
  prefix = prefix_;
  names = names_;
  
  update_prefix();
  update_names();
  update_size();
}



memuse& memuse::operator=(const memuse& m)
{
  bytes = m.get_bytes();
  void update_size();
  
  return *this;
}



inline void memuse::swap_prefix()
{
  if (prefix == "IEC")
    prefix = "SI";
  else // if (prefix == "SI")
    prefix = "IEC";
  
  update_prefix();
  update_size();
}



inline void memuse::swap_names()
{
  if (names == "short")
    names = "long";
  else // if (names == "long")
    names = "short";
  
  update_names();
}



inline void memuse::print()
{
  printf("%s\n", print_str().c_str());
}



inline std::string memuse::print_str()
{
  std::string s;
  s.resize(128);
  
  int pos = sprintf(s.data(), "%.3f", size);
  sprintf(s.data() + pos, " %s", units_table[prefix_indx][names_indx][size_indx].c_str());
  
  return s;
}



inline void memuse::howbig(uint64_t m, uint64_t n, int bytes_per_val)
{
  bytes = (uint64_t) m * n * bytes_per_val;
  update_size();
}



inline void memuse::update_prefix()
{
  if (prefix == "IEC")
  {
    prefix_indx = 0;
    denom = 1024;
  }
  else
  {
    prefix_indx = 1;
    denom = 1000;
  }
}



inline void memuse::update_names()
{
  if (names == "short")
    names_indx = 0;
  else
    names_indx = 1;
}



inline void memuse::update_size()
{
  size = bytes;
  size_indx = 0;
  while (size > denom)
  {
    size /= denom;
    size_indx++;
  }
  
  size_indx = std::min<int>(size_indx, units_table[prefix_indx][names_indx].size() - 1);
}


#endif
