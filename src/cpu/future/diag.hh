// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_FUTURE_DIAG_H
#define FML_CPU_FUTURE_DIAG_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../cpumat.hh"
#include "../cpuvec.hh"


namespace diag
{
  template <typename REAL>
  bool is_diag(const cpumat<REAL> &x)
  {
    if (!x.is_square())
      return false;
    
    const len_t m = x.nrows();
    x_d = x.data_ptr();
    for (len_t j=0; j<x.ncols(); j++)
    {
      for (len_t i=0; i<m; i++)
      {
        if (i == j)
          continue;
        
        if (fabs(x_d[i + m*j]) > (REAL)0.0)
          return false;
      }
    }
    
    return true;
  }
  
  
  template <typename REAL>
  void make_diag(cpumat<REAL> &x, const cpuvec<REAL> &v)
  {

  }
}


#endif
