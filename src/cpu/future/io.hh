// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_CUTURE_IO_H
#define FML_CPU_CUTURE_IO_H
#pragma once


#include <cstdio>
#include <stdexcept>

#include "../cpumat.hh"


namespace io
{
  namespace
  {
    static const int HEADER_LEN = 10;
    static const char *header = "FMLCPUMAT";
  }
  
  
  
  template <typename REAL>
  void write_cpu(const char *filename, const cpumat<REAL> &x)
  {
    FILE *f;
    f = fopen(filename, "wb");
    if (!f)
      throw std::runtime_error("unable to open file for writing");
    
    len_t m = x.nrows();
    len_t n = x.ncols();
    size_t len = (size_t) m*n;
    
    fwrite(header, sizeof(char), HEADER_LEN, f);
    fwrite(&m, sizeof(len_t), 1, f);
    fwrite(&n, sizeof(len_t), 1, f);
    fwrite(x.data_ptr(), sizeof(REAL), len, f);
    fclose(f);
  }
  
  
  
  static inline long int read_offset()
  {
    return (long int) HEADER_LEN + 2*sizeof(len_t);
  }
  
  
  
  template <typename REAL>
  void read_cpu(const char *filename, cpumat<REAL> &x)
  {
    FILE *f;
    f = fopen(filename, "rb");
    if (!f)
      throw std::runtime_error("unable to open file for reading");
    
    char tmp[HEADER_LEN];
    int nread = fread(tmp, sizeof(char), HEADER_LEN, f);
    if (nread != HEADER_LEN || strcmp(tmp, header))
      throw std::runtime_error("mal-formed file");
    
    len_t m, n;
    nread = fread(&m, sizeof(len_t), 1, f);
    nread = fread(&n, sizeof(len_t), 1, f);
    
    x.resize(m, n);
    size_t len = (size_t) m*n;
    
    nread = fread(x.data_ptr(), sizeof(REAL), len, f);
    fclose(f);
  }
  
  
  template <typename REAL>
  void read_cpu_chunk(const char *filename, const len_t row_first, const len_t row_last, cpumat<REAL> &x)
  {
    FILE *f;
    f = fopen(filename, "rb");
    if (!f)
      throw std::runtime_error("unable to open file for reading");
    
    char tmp[HEADER_LEN];
    int nread = fread(tmp, sizeof(char), HEADER_LEN, f);
    if (nread != HEADER_LEN || strcmp(tmp, header))
      throw std::runtime_error("mal-formed file");
    
    len_t m, n;
    nread = fread(&m, sizeof(len_t), 1, f);
    nread = fread(&n, sizeof(len_t), 1, f);
    
    if (row_first > row_last || row_first < 0 || row_last < 0 || row_first >= m || row_last >= m)
      throw std::runtime_error("bad first/last row");
    
    const len_t m_chunk = row_last - row_first + 1;
    x.resize(m_chunk, n);
    
    for (len_t j=0; j<n; j++)
    {
      fseek(f, (row_first + m*j)*sizeof(REAL) + read_offset(), SEEK_SET);
      nread = fread(x.data_ptr() + m_chunk*j, sizeof(REAL), m_chunk, f);
    }
    
    fclose(f);
  }
}


#endif
