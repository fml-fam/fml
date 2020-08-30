// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_LINALG_BLAS_H
#define FML_PAR_GPU_LINALG_BLAS_H
#pragma once


#include "../../../gpu/linalg/linalg_blas.hh"

#include "../parmat.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the product of a distributed and a non-distributed matrix
    whose result is distributed, or the transpose of a distributed matrix with
    a distributed matrix whose result is non-distributed.
    
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    @param[out] ret The product.
    
    @except If x and y are inappropriately sized for a matrix product, the
    method will throw a 'runtime_error' exception.
    
    @impl Uses the BLAS function `Xgemm()`.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void matmult(const parmat_gpu<REAL> &x, const gpumat<REAL> &y,
    parmat_gpu<REAL> &ret)
  {
    err::check_card(x, y, ret);
    
    linalg::matmult(false, false, (REAL)1.0, x.data_obj(), y, ret.data_obj());
  }
  
  /// \overload
  template <typename REAL>
  parmat_gpu<REAL> matmult(const parmat_gpu<REAL> &x, const gpumat<REAL> &y)
  {
    parmat_gpu<REAL> ret(x.get_comm(), x.get_card(), x.nrows(), x.ncols(), x.nrows_before());
    matmult(x, y, ret);
  }
  
  /// \overload
  template <typename REAL>
  void matmult(const parmat_gpu<REAL> &x, const parmat_gpu<REAL> &y,
    gpumat<REAL> &ret)
  {
    err::check_card(x, y, ret);
    
    linalg::matmult(true, false, (REAL)1.0, x.data_obj(), y.data_obj(), ret);
    x.get_comm().allreduce(ret.nrows()*ret.ncols(), ret.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> matmult(const parmat_gpu<REAL> &x, const parmat_gpu<REAL> &y)
  {
    gpumat<REAL> ret(x.get_card(), x.ncols(), y.ncols());
    matmult(x, y, ret);
  }
  
  
  
  /**
    @brief Computes lower triangle of alpha*x^T*x
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the BLAS function `Xsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const parmat_gpu<REAL> &x, gpumat<REAL> &ret)
  {
    err::check_card(x, ret);
    
    const len_t n = x.ncols();
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    linalg::crossprod(alpha, x.data_obj(), ret);
    
    comm r = x.get_comm();
    r.allreduce(n*n, ret.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> crossprod(const REAL alpha, const parmat_gpu<REAL> &x)
  {
    const len_t n = x.ncols();
    gpumat<REAL> ret(x.get_card(), n, n);
    
    crossprod(alpha, x, ret);
    return ret;
  }
}
}


#endif
