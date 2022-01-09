// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_CROSSPROD_H
#define FML_GPU_LINALG_CROSSPROD_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../gpumat.hh"

#include "internals/err.hh"
#include "matmult.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes lower triangle of alpha*x^T*x
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the cuBLAS function `cublasXsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const gpumat<REAL> &x, gpumat<REAL> &ret)
  {
    err::check_card(x, ret);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    matmult(true, false, alpha, x, x, ret);
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> crossprod(const REAL alpha, const gpumat<REAL> &x)
  {
    const len_t n = x.ncols();
    gpumat<REAL> ret(x.get_card(), n, n);
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
    @brief Computes lower triangle of alpha*x*x^T
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the cuBLAS function `cublasXsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void tcrossprod(const REAL alpha, const gpumat<REAL> &x, gpumat<REAL> &ret)
  {
    err::check_card(x, ret);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != ret.nrows() || m != ret.ncols())
      ret.resize(m, m);
    
    matmult(false, true, alpha, x, x, ret);
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> tcrossprod(const REAL alpha, const gpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    gpumat<REAL> ret(x.get_card(), m, m);
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
}
}


#endif
