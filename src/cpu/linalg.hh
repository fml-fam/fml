// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_H
#define FML_CPU_LINALG_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../_internals/linalgutils.hh"
#include "../_internals/omp.hh"

#include "internals/lapack.hh"

#include "cpuhelpers.hh"
#include "cpumat.hh"
#include "cpuvec.hh"


/// @brief Linear algebra functions.
namespace linalg
{
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
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const cpumat<REAL> &x, const cpumat<REAL> &y, cpumat<REAL> &ret)
  {
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
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
  cpumat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    cpumat<REAL> ret(m, n);
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
  
  
  
  /**
    @brief Returns alpha*op(x)*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    
    @except If x and y are inappropriately sized for a matrix product, the
    method will throw a 'runtime_error' exception.
    
    @impl Uses the BLAS function `Xgemm()`.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  cpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    int m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    fml::linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    cpumat<REAL> ret(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
    
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
  void matmult(const bool transx, const bool transy, const REAL alpha, const cpumat<REAL> &x, const cpumat<REAL> &y, cpumat<REAL> &ret)
  {
    int m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    fml::linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
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
    fml::lapack::syrk('L', 'T', n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
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
    fml::lapack::syrk('L', 'N', m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
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
    
    #pragma omp parallel for shared(tx) schedule(dynamic, 1) if(m*n>fml::omp::OMP_MIN_SIZE)
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
  
  
  
  /**
    @brief Computes the PLU factorization with partial pivoting.
    
    @details The input is replaced by its LU factorization, with L
    unit-diagonal.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] p Vector of pivots, representing the diagonal matrix P in the
    PLU.
    @param[out] info The LAPACK return number.
    
    @impl Uses the LAPACK function `Xgetrf()`.
    
    @allocs If the pivot vector is inappropriately sized, it will automatically
    be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lu(cpumat<REAL> &x, cpuvec<int> &p, int &info)
  {
    info = 0;
    const len_t m = x.nrows();
    const len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    fml::lapack::getrf(m, x.ncols(), x.data_ptr(), m, p.data_ptr(), &info);
  }
  
  /// \overload
  template <typename REAL>
  void lu(cpumat<REAL> &x)
  {
    cpuvec<int> p;
    int info;
    
    lu(x, p, info);
    
    fml::linalgutils::check_info(info, "getrf");
  }
  
  
  
  /**
    @brief Computes the trace, i.e. the sum of the diagonal.
    
    @param[in] x Input data matrix.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL trace(const cpumat<REAL> &x)
  {
    const REAL *x_d = x.data_ptr();
    const len_t m = x.nrows();
    const len_t minmn = std::min(m, x.ncols());
    
    REAL tr = 0;
    #pragma omp simd reduction(+:tr)
    for (len_t i=0; i<minmn; i++)
      tr += x_d[i + i*m];
    
    return tr;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int svd_internals(const int nu, const int nv, cpumat<REAL> &x, cpuvec<REAL> &s,
      cpumat<REAL> &u, cpumat<REAL> &vt)
    {
      int info = 0;
      char jobz;
      int ldvt;
      
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      s.resize(minmn);
      
      if (nu == 0 && nv == 0)
      {
        jobz = 'N';
        ldvt = 1; // value is irrelevant, but must exist!
      }
      else if (nu <= minmn && nv <= minmn)
      {
        jobz = 'S';
        ldvt = minmn;
        
        u.resize(m, minmn);
        vt.resize(minmn, n);
      }
      else
      {
        jobz = 'A';
        ldvt = n;
      }
      
      cpuvec<int> iwork(8*minmn);
      
      REAL tmp;
      fml::lapack::gesdd(jobz, m, n, x.data_ptr(), m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), ldvt, &tmp, -1, iwork.data_ptr(), &info);
      int lwork = (int) tmp;
      cpuvec<REAL> work(lwork);
      
      fml::lapack::gesdd(jobz, m, n, x.data_ptr(), m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), ldvt, work.data_ptr(), lwork, iwork.data_ptr(), &info);
      
      return info;
    }
  }
  
  /**
    @brief Computes the singular value decomposition.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses the LAPACK function `Xgesvd()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void svd(cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    cpumat<REAL> ignored;
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    fml::linalgutils::check_info(info, "gesdd");
  }
  
  /// \overload
  template <typename REAL>
  void svd(cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    int info = svd_internals(1, 1, x, s, u, vt);
    fml::linalgutils::check_info(info, "gesdd");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, cpumat<REAL> &x,
      cpuvec<REAL> &values, cpumat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      int info = 0;
      int nfound;
      char jobz;
      
      len_t n = x.nrows();
      values.resize(n);
      cpuvec<int> support;
      
      if (only_values)
        jobz = 'N';
      else
      {
        jobz = 'V';
        vectors.resize(n, n);
        support.resize(2*n);
      }
      
      REAL worksize;
      int lwork, liwork;
      fml::lapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), n, (REAL) 0.f, (REAL) 0.f,
        0, 0, (REAL) 0.f, &nfound, values.data_ptr(), vectors.data_ptr(), n,
        support.data_ptr(), &worksize, -1, &liwork, -1,
        &info);
      
      lwork = (int) worksize;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      fml::lapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), n, (REAL) 0.f, (REAL) 0.f,
        0, 0, (REAL) 0.f, &nfound, values.data_ptr(), vectors.data_ptr(), n,
        support.data_ptr(), work.data_ptr(), lwork, iwork.data_ptr(), liwork,
        &info);
      
      return info;
    }
  }
  
  /**
    @brief Compute the eigenvalues and optionally the eigenvectors for a
    symmetric matrix.
    
    @details The input data is overwritten.
    
    @param[inout] x Input data matrix. Should be square.
    @param[out] values Eigenvalues.
    @param[out] vectors Eigenvectors.
    
    @impl Uses the LAPACK functions `Xsyevr()`.
    
    @allocs If any output's dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void eigen_sym(cpumat<REAL> &x, cpuvec<REAL> &values)
  {
    cpumat<REAL> ignored;
    
    int info = eig_sym_internals(true, x, values, ignored);
    fml::linalgutils::check_info(info, "syevr");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(cpumat<REAL> &x, cpuvec<REAL> &values, cpumat<REAL> &vectors)
  {
    int info = eig_sym_internals(false, x, values, vectors);
    fml::linalgutils::check_info(info, "syevr");
  }
  
  
  
  /**
    @brief Compute the matrix inverse.
    
    @details The input is replaced by its inverse, computed via a PLU.
    
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the LAPACK functions `Xgetrf()` (LU) and `Xgetri()` (solve).
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void invert(cpumat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    cpuvec<int> p;
    int info;
    lu(x, p, info);
    fml::linalgutils::check_info(info, "getrf");
    
    // Invert
    REAL tmp;
    fml::lapack::getri(n, x.data_ptr(), n, p.data_ptr(), &tmp, -1, &info);
    int lwork = (int) tmp;
    cpuvec<REAL> work(lwork);
    
    fml::lapack::getri(n, x.data_ptr(), n, p.data_ptr(), work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "getri");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void solver(cpumat<REAL> &x, len_t ylen, len_t nrhs, REAL *y_d)
    {
      const len_t n = x.nrows();
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      if (n != ylen)
        throw std::runtime_error("rhs 'y' must be compatible with data matrix 'x'");
      
      int info;
      cpuvec<int> p(n);
      fml::lapack::gesv(n, nrhs, x.data_ptr(), n, p.data_ptr(), y_d, n, &info);
      fml::linalgutils::check_info(info, "gesv");
    }
  }
  
  /**
    @brief Solve a system of equations.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input LHS. Should be square. Overwritten by LU.
    @param[inout] y Input RHS. Overwritten by solution.
    
    @impl Uses the LAPACK functions `Xgesv()`.
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square or if the RHS is incompatible with the
    LHS, a `runtime_error` exception is thrown. If an allocation fails, a
    `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void solve(cpumat<REAL> &x, cpuvec<REAL> &y)
  {
    solver(x, y.size(), 1, y.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  void solve(cpumat<REAL> &x, cpumat<REAL> &y)
  {
    solver(x, y.nrows(), y.ncols(), y.data_ptr());
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void qr_internals(const bool pivot, cpumat<REAL> &x, cpuvec<REAL> &qraux, cpuvec<REAL> &work)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      int info = 0;
      qraux.resize(minmn);
      
      REAL tmp;
      if (pivot)
        fml::lapack::geqp3(m, n, NULL, m, NULL, NULL, &tmp, -1, &info);
      else
        fml::lapack::geqrf(m, n, NULL, m, NULL, &tmp, -1, &info);
      
      int lwork = std::max((int) tmp, 1);
      if (lwork > work.size())
        work.resize(lwork);
      
      if (pivot)
      {
        cpuvec<int> p(n);
        p.fill_zero();
        fml::lapack::geqp3(m, n, x.data_ptr(), m, p.data_ptr(), qraux.data_ptr(), work.data_ptr(), lwork, &info);
      }
      else
        fml::lapack::geqrf(m, n, x.data_ptr(), m, qraux.data_ptr(), work.data_ptr(), lwork, &info);
      
      if (info != 0)
      {
        if (pivot)
          fml::linalgutils::check_info(info, "geqp3");
        else
          fml::linalgutils::check_info(info, "geqrf");
      }
    }
  }
  
  /**
    @brief Computes the QR decomposition.
    
    @details The factorization works mostly in-place by modifying the input
    data. After execution, the matrix will be the LAPACK-like compact QR
    representation.
    
    @param[in] pivot Should the factorization use column pivoting?
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] qraux Auxiliary data for compact QR.
    
    @impl Uses the LAPACK function `Xgeqp3()` if pivoting and `Xgeqrf()`
    otherwise.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr(const bool pivot, cpumat<REAL> &x, cpuvec<REAL> &qraux)
  {
    cpuvec<REAL> work;
    qr_internals(pivot, x, qraux, work);
  }
  
  /**
    @brief Recover the Q matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[in] qraux Auxiliary data for compact QR.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl Uses the LAPACK function `Xormqr()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_Q(const cpumat<REAL> &QR, const cpuvec<REAL> &qraux, cpumat<REAL> &Q, cpuvec<REAL> &work)
  {
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    int info = 0;
    REAL tmp;
    fml::lapack::ormqr('L', 'N', m, minmn, n, QR.data_ptr(), m, qraux.data_ptr(),
      Q.data_ptr(), m, &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    Q.resize(m, n);
    Q.fill_eye();
    
    fml::lapack::ormqr('L', 'N', m, minmn, n, QR.data_ptr(), m, qraux.data_ptr(),
      Q.data_ptr(), m, work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "ormqr");
  }
  
  /**
    @brief Recover the R matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[out] R The R matrix.
    
    @impl Uses the LAPACK function `Xlacpy()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_R(const cpumat<REAL> &QR, cpumat<REAL> &R)
  {
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    
    R.resize(n, n);
    R.fill_zero();
    fml::lapack::lacpy('U', m, n, QR.data_ptr(), m, R.data_ptr(), n);
  }
  
  
  
  
  
  /**
    @brief Computes the singular value decomposition for tall/skinny data.
    The number of rows must be greater than the number of columns. If the number
    of rows is not significantly larger than the number of columns, this may not
    be more efficient than simply calling `linalg::svd()`.
    
    @details The operation works by computing a QR and then taking the SVD of
    the R matrix. The left singular vectors are Q times the left singular
    vectors from R's SVD, and the singular value and the right singular vectors
    are those from R's SVD.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `linalg::qr()` and `linalg::svd()`, and if computing the
    left/right singular vectors, `linalg::qr_R()` and `linalg::qr_Q()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void tssvd(cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    cpuvec<REAL> qraux(n);
    cpuvec<REAL> work(m);
    qr_internals(false, x, qraux, work);
    
    cpumat<REAL> R(n, n);
    qr_R(x, R);
    
    cpumat<REAL> u_R(n, n);
    svd(R, s, u_R, vt);
    
    u.resize(m, n);
    u.fill_eye();
    
    qr_Q(x, qraux, u, work);
    
    matmult(false, false, (REAL)1.0, u, u_R, x);
    cpuhelpers::cpu2cpu(x, u);
  }
  
  /// \overload
  template <typename REAL>
  void tssvd(cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    s.resize(n);
    
    cpuvec<REAL> qraux(n);
    cpuvec<REAL> work(m);
    qr_internals(false, x, qraux, work);
    
    REAL *x_d = x.data_ptr();
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=j+1; i<n; i++)
        x_d[i + m*j] = (REAL) 0.0;
    }
    
    int info = 0;
    cpuvec<int> iwork(8*n);
    
    REAL tmp;
    fml::lapack::gesdd('N', n, n, x_d, m, s.data_ptr(), NULL, m, NULL, 1, &tmp, -1, iwork.data_ptr(), &info);
    int lwork = (int) tmp;
    work.resize(lwork);
    
    fml::lapack::gesdd('N', n, n, x_d, m, s.data_ptr(), NULL, m, NULL, 1, work.data_ptr(), lwork, iwork.data_ptr(), &info);
    fml::linalgutils::check_info(info, "gesdd");
  }
  
  
  
  /**
    @brief Computes the singular value decomposition using the
    "crossproducts SVD". This method is not numerically stable.
    
    @details The operation works by computing the crossproducts matrix X^T * X
    and then computing the eigenvalue decomposition. 
    
    @param[inout] x Input data matrix.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `linalg::qr()` and `linalg::svd()`, and if computing the
    left/right singular vectors, `linalg::qr_R()` and `linalg::qr_Q()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void cpsvd(const cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const len_t minmn = std::min(m, n);
    
    cpumat<REAL> cp;
    
    if (m >= n)
    {
      crossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, vt);
      vt.rev_cols();
      cpuhelpers::cpu2cpu(vt, cp);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, u);
      u.rev_cols();
      cpuhelpers::cpu2cpu(u, cp);
    }
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
    
    REAL *ev_d;
    if (m >= n)
      ev_d = vt.data_ptr();
    else
      ev_d = cp.data_ptr();
    
    #pragma omp for simd
    for (len_t j=0; j<minmn; j++)
    {
      for (len_t i=0; i<minmn; i++)
        ev_d[i + minmn*j] /= s_d[j];
    }
    
    if (m >= n)
    {
      matmult(false, false, (REAL)1.0, x, vt, u);
      xpose(cp, vt);
    }
    else
      matmult(true, false, (REAL)1.0, cp, x, vt);
  }
  
  /// \overload
  template <typename REAL>
  void cpsvd(const cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const len_t minmn = std::min(m, n);
    
    cpumat<REAL> cp;
    
    if (m >= n)
    {
      crossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s);
    }
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
  }
}


#endif
