// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_CROSSPROD_H
#define FML_MPI_LINALG_CROSSPROD_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../mpimat.hh"

#include "internals/err.hh"
#include "internals/pblas.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes lower triangle of alpha*x^T*x
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the BLAS function `pXsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    err::check_grid(x, ret);
    
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    ret.fill_zero();
    fml::pblas::syrk('L', 'T', n, x.nrows(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> crossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    const len_t n = x.ncols();
    const grid g = x.get_grid();
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
    @brief Computes lower triangle of alpha*x*x^T
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the PBLAS function `pXsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void tcrossprod(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    err::check_grid(x, ret);
    
    const len_t m = x.nrows();
    
    if (m != ret.nrows() || m != ret.ncols())
      ret.resize(m, m);
    
    ret.fill_zero();
    fml::pblas::syrk('L', 'N', m, x.ncols(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> tcrossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    const len_t n = x.nrows();
    const grid g = x.get_grid();
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
}
}


#endif
