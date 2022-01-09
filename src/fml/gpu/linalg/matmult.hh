// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_MATMULT_H
#define FML_GPU_LINALG_MATMULT_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../gpumat.hh"

#include "internals/err.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes ret = alpha*op(x)*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    @param[out] ret The product.
    
    @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception.
    
    @impl Uses the cuBLAS function `cublasXgemm()`.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha,
    const gpumat<REAL> &x, const gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    err::check_card(x, y, ret);
    
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    int m, n, k;
    fml::linalgutils::matmult_params(transx, transy, mx, x.ncols(),
      my, y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::gemm(x.get_card()->blas_handle(),
      cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(),
      my, (REAL)0, ret.data_ptr(), m);
    gpublas::err::check_ret(check, "gemm");
  }
  
  
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha,
    const gpumat<REAL> &x, const gpumat<REAL> &y)
  {
    gpumat<REAL> ret(x.get_card());
    matmult(transx, transy, alpha, x, y, ret);
    
    return ret;
  }
  
  
  
  /// \overload
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha,
    const gpumat<REAL> &x, const gpuvec<REAL> &y, gpuvec<REAL> &ret)
  {
    err::check_card(x, y, ret);
    
    const len_t mx = x.nrows();
    const len_t my = y.size();
    
    int m, n, k;
    fml::linalgutils::matmult_params(transx, transy, mx, x.ncols(),
      my, 1, &m, &n, &k);
    auto c = x.get_card();
    int len = std::max(m, n);
    if (len != ret.size())
      ret.resize(len);
    
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::gemm(c->blas_handle(), cbtransx, cbtransy,
      m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0,
      ret.data_ptr(), m);
    gpublas::err::check_ret(check, "gemm");
  }
  
  
  
  /// \overload
  template <typename REAL>
  gpuvec<REAL> matmult(const bool transx, const bool transy, const REAL alpha,
    const gpumat<REAL> &x, const gpuvec<REAL> &y)
  {
    gpuvec<REAL> ret(x.get_card());
    matmult(transx, transy, alpha, x, y, ret);
    
    return ret;
  }
  
  
  
  /// \overload
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha,
    const gpuvec<REAL> &x, const gpumat<REAL> &y, gpuvec<REAL> &ret)
  {
    err::check_card(x, y, ret);
    
    const len_t mx = x.size();
    const len_t my = y.nrows();
    
    int m, n, k;
    fml::linalgutils::matmult_params(transx, transy, mx, 1,
      my, y.ncols(), &m, &n, &k);
    auto c = x.get_card();
    int len = std::max(m, n);
    if (len != ret.size())
      ret.resize(len);
    
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::gemm(c->blas_handle(), cbtransx, cbtransy,
      m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0,
      ret.data_ptr(), m);
    gpublas::err::check_ret(check, "gemm");
  }
  
  
  
  /// \overload
  template <typename REAL>
  gpuvec<REAL> matmult(const bool transx, const bool transy, const REAL alpha,
    const gpuvec<REAL> &x, const gpumat<REAL> &y)
  {
    gpuvec<REAL> ret(x.get_card());
    matmult(transx, transy, alpha, x, y, ret);
    
    return ret;
  }
}
}


#endif
