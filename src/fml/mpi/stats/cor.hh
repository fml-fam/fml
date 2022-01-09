// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_STATS_COR_H
#define FML_MPI_STATS_COR_H
#pragma once


#include <stdexcept>

#include "../../_internals/omp.hh"

#include "../mpimat.hh"
#include "../dimops.hh"

#include "../linalg/crossprod.hh"
#include "../linalg/matmult.hh"


namespace fml
{
namespace stats
{
  /**
    @brief Covariance.
    
    @details Computes the lower triangle of the Pearson correlation matrix.
    Centering is done in-place.
    
    @param[inout] x,y Input data. For the matrix variants, data is mean-centered
    on return.
    @param[out] cov The correlation matrix.
    
    @except In the matrix-matrix and vector-vector variants, if the object
    dimensions/sizes are non-conformable, a `runtime_error` exception is thrown.
    
    @tparam REAL should be 'float' or 'double'.
  */
  template <typename REAL>
  void cor(mpimat<REAL> &x, mpimat<REAL> &cov)
  {
    dimops::scale(true, true, x);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::crossprod(alpha, x, cov);
  }
  
  
  
  /// \overload
  template <typename REAL>
  void cor(mpimat<REAL> &x, mpimat<REAL> &y, mpimat<REAL> &cov)
  {
    if (x.nrows() != y.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    dimops::scale(true, true, x);
    dimops::scale(true, true, y);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::matmult(true, false, alpha, x, y, cov);
  }
}
}


#endif
