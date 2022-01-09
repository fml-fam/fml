// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_MATMULT_H
#define FML_CPU_LINALG_MATMULT_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../cpumat.hh"

#include "internals/blas.hh"


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
    
    @impl Uses the BLAS function `Xgemm()`.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha,
    const cpumat<REAL> &x, const cpumat<REAL> &y, cpumat<REAL> &ret)
  {
    len_t m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    fml::linalgutils::matmult_params(transx, transy, mx, x.ncols(), my,
      y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::blas::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), mx, y.data_ptr(), my,
      (REAL)0, ret.data_ptr(), m);
  }
  
  
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha,
    const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    cpumat<REAL> ret;
    matmult(transx, transy, alpha, x, y, ret);
    
    return ret;
  }
  
  
  
  /// \overload
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha,
    const cpumat<REAL> &x, const cpuvec<REAL> &y, cpuvec<REAL> &ret)
  {
    len_t m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.size();
    
    fml::linalgutils::matmult_params(transx, transy, mx, x.ncols(), my,
      1, &m, &n, &k);
    
    int len = std::max(m, n);
    if (len != ret.size())
      ret.resize(len);
      
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::blas::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), mx, y.data_ptr(), my,
      (REAL)0, ret.data_ptr(), m);
  }
  
  
  
  /// \overload
  template <typename REAL>
  cpuvec<REAL> matmult(const bool transx, const bool transy, const REAL alpha,
    const cpumat<REAL> &x, const cpuvec<REAL> &y)
  {
    cpuvec<REAL> ret;
    matmult(transx, transy, alpha, x, y, ret);
    
    return ret;
  }
  
  
  
  /// \overload
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha,
    const cpuvec<REAL> &x, const cpumat<REAL> &y, cpuvec<REAL> &ret)
  {
    len_t m, n, k;
    const len_t mx = x.size();
    const len_t my = y.nrows();
    
    fml::linalgutils::matmult_params(transx, transy, mx, 1, my,
      y.ncols(), &m, &n, &k);
    
    int len = std::max(m, n);
    if (len != ret.size())
      ret.resize(len);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::blas::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), mx, y.data_ptr(), my,
      (REAL)0, ret.data_ptr(), m);
  }
  
  
  
  /// \overload
  template <typename REAL>
  cpuvec<REAL> matmult(const bool transx, const bool transy, const REAL alpha,
    const cpuvec<REAL> &x, const cpumat<REAL> &y)
  {
    cpuvec<REAL> ret;
    matmult(transx, transy, alpha, x, y, ret);
    
    return ret;
  }
}
}


#endif
