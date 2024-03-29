// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_STATS_COV_H
#define FML_CPU_STATS_COV_H
#pragma once


#include <stdexcept>

#include "../../_internals/omp.hh"

#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "../dimops.hh"

#include "../linalg/crossprod.hh"
#include "../linalg/matmult.hh"


namespace fml
{
namespace stats
{
  /**
    @brief Covariance.
    
    @details Computes the lower triangle of the variance-covariance matrix.
    Centering is done in-place.
    
    @param[inout] x,y Input data. For the matrix variants, data is mean-centered
    on return.
    @param[out] cov The covariance matrix.
    
    @except In the matrix-matrix and vector-vector variants, if the object
    dimensions/sizes are non-conformable, a `runtime_error` exception is thrown.
    
    @tparam REAL should be 'float' or 'double'.
  */
  template <typename REAL>
  void cov(cpumat<REAL> &x, cpumat<REAL> &cov)
  {
    dimops::scale(true, false, x);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::crossprod(alpha, x, cov);
  }
  
  
  
  /// \overload
  template <typename REAL>
  void cov(cpumat<REAL> &x, cpumat<REAL> &y, cpumat<REAL> &cov)
  {
    if (x.nrows() != y.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    dimops::scale(true, false, x);
    dimops::scale(true, false, y);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::matmult(true, false, alpha, x, y, cov);
  }
  
  
  
  /// \overload
  template <typename REAL>
  REAL cov(const cpuvec<REAL> &x, const cpuvec<REAL> &y)
  {
    const len_t n = x.size();
    if (n != y.size())
      throw std::runtime_error("non-conformal arguments");
    
    const REAL *x_d = x.data_ptr();
    const REAL *y_d = y.data_ptr();
    
    REAL sum_xy = 0, sum_x = 0, sum_y = 0;
    
    #pragma omp parallel for reduction(+: sum_xy, sum_x, sum_y) if(n>fml::omp::OMP_MIN_SIZE)
    for (len_t i=0; i<n; i++)
    {
      const REAL xi = x_d[i];
      const REAL yi = y_d[i];
      
      sum_xy += xi*yi;
      sum_x += xi;
      sum_y += yi;
    }
    
    return (sum_xy - (sum_x*sum_y*((REAL) 1./n))) * ((REAL) 1./(n-1));
  }
}
}


#endif
