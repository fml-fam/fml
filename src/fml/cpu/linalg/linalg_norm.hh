// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_NORM_H
#define FML_CPU_LINALG_LINALG_NORM_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "lapack.hh"
#include "linalg_factorizations.hh"


namespace fml
{
/// @brief Linear algebra functions.
namespace linalg
{
  /**
    @brief Computes the 1 matrix norm, the maximum absolute column sum.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_1(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    for (len_t j=0; j<n; j++)
    {
      REAL tmp = 0;
      for (len_t i=0; i<m; i++)
        tmp += fabsf(x_d[i + m*j]);
      
      if (tmp > norm)
        norm = tmp;
    }
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the infinity matrix norm, the maximum absolute row sum.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to store the row sums.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_I(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    cpuvec<REAL> tmp(m);
    tmp.fill_zero();
    REAL *tmp_d = tmp.data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m; i++)
        tmp_d[i] += fabsf(x_d[i + m*j]);
    }
    
    for (len_t i=0; i<m; i++)
    {
      if (tmp_d[i] > norm)
        norm = tmp_d[i];
    }
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the Frobenius/Euclidean matrix norm.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_F(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL scale = 0;
    REAL sumsq = 1;
    
    for (len_t j=0; j<n; j++)
      lapack::lassq(m, x_d + m*j, 1, &scale, &sumsq);
    
    return scale * sqrtf(sumsq);
  }
  
  
  
  /**
    @brief Computes the maximum modulus matrix norm.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_M(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m; i++)
      {
        REAL tmp = fabsf(x_d[i + m*j]);
        if (tmp > norm)
          norm = tmp;
      }
    }
    
    return norm;
  }
}
}


#endif
