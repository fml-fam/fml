// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_FUTURE_LINALG_H
#define FML_CPU_FUTURE_LINALG_H
#pragma once


#include "../linalg.hh"


namespace linalg
{
  template <typename REAL>
  void det(cpumat<REAL> &x, int &sign, REAL &modulus)
  {
    const len_t m = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    cpuvec<int> p;
    int info;
    lu(x, p, info);
    
    if (info != 0)
    {
      if (info > 0)
      {
        sign = 1;
        modulus = -INFINITY;
        return;
      }
      else
        return;
    }
    
    
    // get determinant
    REAL mod = 0.0;
    int sgn = 1;
    
    const int *ipiv = p.data_ptr();
    for (int i=0; i<m; i++)
    {
      if (ipiv[i] != (i + 1))
        sgn = -sgn;
    }
    
    const REAL *a = x.data_ptr();
    
    #pragma omp parallel for reduction(+:mod) reduction(*:sgn)
    for (int i=0; i<m; i+=m+1)
    {
      const REAL d = a[i + m*i];
      if (d < 0)
      {
        mod += log(-d);
        sgn *= -1;
      }
      else
        mod += log(d);
    }
    
    modulus = mod;
    sign = sgn;
  }
  
}


#endif
