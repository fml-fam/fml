// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_BLAS_H
#define FML_CPU_LINALG_LINALG_BLAS_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "blas.hh"

#include "../cpumat.hh"


namespace fml
{
/// @brief Linear algebra functions.
namespace linalg
{
  /**
    @brief Computes the dot product of two vectors, i.e. the sum of the product
    of the elements.
    
    @details NOTE: if the vectors are of different length, the dot product will
    use only the indices of the smaller-sized vector.
    
    @param[in] x,y Vectors.
    
    @return The dot product.
    
    @tparam REAL should be 'float' or 'double' ('int' is also ok).
   */
  template <typename REAL>
  REAL dot(const cpuvec<REAL> &x, const cpuvec<REAL> &y)
  {
    const len_t n = std::min(x.size(), y.size());
    const REAL *x_d = x.data_ptr();
    const REAL *y_d = y.data_ptr();
    
    REAL d = 0;
    #pragma omp simd reduction(+:d)
    for (len_t i=0; i<n; i++)
      d += x_d[i] * y_d[i];
    
    return d;
  }
  
  template <typename REAL>
  REAL dot(const cpuvec<REAL> &x)
  {
    return dot(x, x);
  }
  
  
  
  /**
    @brief Returns alpha*op(x) + beta*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha,beta Scalars.
    @param[in] x,y The inputs to the sum.
    @param[out] ret The sum.
    
    @except If x and y are inappropriately sized for the sum, the method will
    throw a 'runtime_error' exception.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha,
    const REAL beta, const cpumat<REAL> &x, const cpumat<REAL> &y,
    cpumat<REAL> &ret)
  {
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(),
      y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    const REAL *x_d = x.data_ptr();
    const REAL *y_d = y.data_ptr();
    REAL *ret_d = ret.data_ptr();
    
    if (!transx && !transy)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          ret_d[i + m*j] = alpha*x_d[i + m*j] + beta*y_d[i + m*j];
      }
    }
    else if (transx && transy)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<m; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<n; i++)
          ret_d[j + m*i] = alpha*x_d[i + n*j] + beta*y_d[i + n*j];
      }
    }
    else if (transx && !transy)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          ret_d[i + m*j] = alpha*x_d[j + n*i] + beta*y_d[i + m*j];
      }
    }
    else if (!transx && transy)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          ret_d[i + m*j] = alpha*x_d[i + m*j] + beta*y_d[j + n*i];
      }
    }
  }
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> add(const bool transx, const bool transy, const REAL alpha,
    const REAL beta, const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(),
      y.nrows(), y.ncols(), &m, &n);
    
    cpumat<REAL> ret(m, n);
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
  
  
  
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
  void crossprod(const REAL alpha, const cpumat<REAL> &x, cpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    ret.fill_zero();
    fml::blas::syrk('L', 'T', n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
  }
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> crossprod(const REAL alpha, const cpumat<REAL> &x)
  {
    const len_t n = x.ncols();
    cpumat<REAL> ret(n, n);
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
    @brief Computes lower triangle of alpha*x*x^T
    
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
  void tcrossprod(const REAL alpha, const cpumat<REAL> &x, cpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != ret.nrows() || m != ret.ncols())
      ret.resize(m, m);
    
    ret.fill_zero();
    fml::blas::syrk('L', 'N', m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
  }
  
  template <typename REAL>
  cpumat<REAL> tcrossprod(const REAL alpha, const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    cpumat<REAL> ret(m, m);
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
    @brief Computes the transpose out-of-place (i.e. in a copy).
    
    @param[in] x Input data matrix.
    @param[out] tx The transpose.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void xpose(const cpumat<REAL> &x, cpumat<REAL> &tx)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      tx.resize(n, m);
    
    const int blocksize = 8;
    const REAL *x_d = x.data_ptr();
    REAL *tx_d = tx.data_ptr();
    
    #pragma omp parallel for shared(tx) schedule(dynamic, 1) if(m*n > fml::omp::OMP_MIN_SIZE)
    for (int j=0; j<n; j+=blocksize)
    {
      for (int i=0; i<m; i+=blocksize)
      {
        for (int col=j; col<j+blocksize && col<n; ++col)
        {
          for (int row=i; row<i+blocksize && row<m; ++row)
            tx_d[col + n*row] = x_d[row + m*col];
        }
      }
    }
  }
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> xpose(const cpumat<REAL> &x)
  {
    cpumat<REAL> tx(x.ncols(), x.nrows());
    xpose(x, tx);
    return tx;
  }
}
}


#endif
