// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_UTILS_H
#define FML_CPU_UTILS_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../internals/lapack.hh"

#include "../cpumat.hh"


namespace utils
{
  template <typename REAL>
  bool is_symmetric(const cpumat<REAL> &x)
  {
    if (!x.is_square())
      return false;
    
    const int blocksize = 8;
    const len_t n = x.nrows();
    
    for (len_t j=0; j<n; j+=blocksize)
    {
      for (len_t i=j; i<n; i+=blocksize)
      {
        for (len_t col=j; col<j+blocksize && col<n; ++col)
        {
          for (len_t row=i; row<i+blocksize && row<n; ++row)
          {
            const bool check = samenum(x[col + n*row], x[row + n*col]);
            if (!check)
              return false;
          }
        }
      }
    }
    
    return true;
  }
  
  
  template <typename REAL>
  void symmetrize(cpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("non-square matrix");
    
    const int blocksize = 8;
    const len_t n = x.nrows();
    
    REAL *x_d = x.data_ptr();
    for (len_t j=0; j<n; j+=blocksize)
    {
      for (len_t i=j+1; i<n; i+=blocksize)
      {
        for (len_t col=j; col<j+blocksize && col<n; ++col)
        {
          for (len_t row=i; row<i+blocksize && row<n; ++row)
            x_d[col + n*row] = x_d[row + n*col];
        }
      }
    }
  }
}


#endif
