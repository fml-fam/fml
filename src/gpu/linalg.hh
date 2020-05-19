// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_H
#define FML_GPU_LINALG_H
#pragma once


#include <stdexcept>
#include <string>

#include "../_internals/linalgutils.hh"

#include "arch/arch.hh"

#include "internals/gpu_utils.hh"
#include "internals/gpuscalar.hh"
#include "internals/kernelfuns.hh"

#include "gpuhelpers.hh"
#include "gpumat.hh"
#include "gpuvec.hh"


namespace linalg
{
  namespace err
  {  
    template <typename REAL, class ARR>
    void check_card(const gpumat<REAL> &a, const ARR &b)
    {
      if (a.get_card()->get_id() != b.get_card()->get_id())
        throw std::runtime_error("gpumat/gpuvex objects must be allocated on the same gpu");
    }
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
    
    @impl Uses the cuBLAS function `cublasXgeam()`.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const gpumat<REAL> &x, const gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    err::check_card(x, y);
    err::check_card(x, ret);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    auto c = x.get_card();
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::geam(c->blas_handle(), cbtransx, cbtransy, m, n, alpha, x.data_ptr(), x.nrows(), beta, y.data_ptr(), y.nrows(), ret.data_ptr(), m);
    gpublas::err::check_ret(check, "geam");
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const gpumat<REAL> &x, const gpumat<REAL> &y)
  {
    err::check_card(x, y);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    auto c = x.get_card();
    gpumat<REAL> ret(c, m, n);
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
    
    @impl Uses the cuBLAS function `cublasXgemm()`.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  gpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const gpumat<REAL> &x, const gpumat<REAL> &y)
  {
    err::check_card(x, y);
    
    int m, n, k;
    fml::linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    auto c = x.get_card();
    gpumat<REAL> ret(c, m, n);
    
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::gemm(c->blas_handle(), cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), x.nrows(), y.data_ptr(), y.nrows(), (REAL)0, ret.data_ptr(), m);
    gpublas::err::check_ret(check, "gemm");
    
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
    
    @impl Uses the cuBLAS function `cublasXgemm()`.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const gpumat<REAL> &x, const gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    err::check_card(x, y);
    err::check_card(x, ret);
    
    int m, n, k;
    fml::linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::gemm(x.get_card()->blas_handle(), cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), x.nrows(), y.data_ptr(), y.nrows(), (REAL)0, ret.data_ptr(), m);
    gpublas::err::check_ret(check, "gemm");
  }
  
  
  
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
    
    ret.fill_zero();
    
    auto cbh = x.get_card()->blas_handle();
    gpublas_operation_t trans = GPUBLAS_OP_T;
    gpublas_fillmode_t uplo = GPUBLAS_FILL_L;
    
    gpublas_status_t check = gpublas::syrk(cbh, uplo, trans, n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
    gpublas::err::check_ret(check, "syrk");
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
    
    ret.fill_zero();
    
    auto cbh = x.get_card()->blas_handle();
    gpublas_operation_t trans = GPUBLAS_OP_N;
    gpublas_fillmode_t uplo = GPUBLAS_FILL_L;
    
    gpublas_status_t check = gpublas::syrk(cbh, uplo, trans, m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
    gpublas::err::check_ret(check, "syrk");
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
  
  
  
  /**
    @brief Computes the transpose out-of-place (i.e. in a copy).
    
    @param[in] x Input data matrix.
    @param[out] tx The transpose.
    
    @impl Uses the cuBLAS function `cublasXgeam()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void xpose(const gpumat<REAL> &x, gpumat<REAL> &tx)
  {
    err::check_card(x, tx);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      tx.resize(n, m);
    
    auto cbh = x.get_card()->blas_handle();
    
    gpublas_status_t check = gpublas::geam(cbh, GPUBLAS_OP_T, GPUBLAS_OP_N, n, m, (REAL)1.0, x.data_ptr(), m, (REAL) 0.0, tx.data_ptr(), n, tx.data_ptr(), n);
    gpublas::err::check_ret(check, "geam");
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> xpose(const gpumat<REAL> &x)
  {
    gpumat<REAL> tx(x.get_card(), x.ncols(), x.nrows());
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
    
    @impl Uses the cuSOLVER function `cusolverDnXgetrf()`.
    
    @allocs If the pivot vector is inappropriately sized, it will automatically
    be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void lu(gpumat<REAL> &x, gpuvec<int> &p, int &info)
  {
    err::check_card(x, p);
    
    info = 0;
    const int m = x.nrows();
    auto c = x.get_card();
    
    const len_t lipiv = std::min(m, x.ncols());
    if (!p.get_card()->valid_card())
      p.inherit(c);
    
    p.resize(lipiv);
    
    int lwork;
    gpulapack_status_t check = gpulapack::getrf_buflen(c->lapack_handle(), m, m, x.data_ptr(), m, &lwork);
    gpulapack::err::check_ret(check, "getrf_bufferSize");
    
    gpuvec<REAL> work(c, lwork);
    gpuscalar<int> info_device(c, info);
    
    check = gpulapack::getrf(c->lapack_handle(), m, m, x.data_ptr(), m, work.data_ptr(), p.data_ptr(), info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "getrf");
  }
  
  /// \overload
  template <typename REAL>
  void lu(gpumat<REAL> &x)
  {
    gpuvec<int> p(x.get_card());
    int info;
    
    lu(x, p, info);
    
    fml::linalgutils::check_info(info, "getrf");
  }
  
  
  
  namespace
  {
    static __global__ void kernel_lu_pivot_sgn(const len_t n, int *ipiv, int *sgn)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      
      if (i < n)
      {
        ipiv[i] = (ipiv[i] != (i+1) ? -1 : 1);
        atomicAdd(sgn, ipiv[i]);
        
        if (threadIdx.x == 0)
          (*sgn) = ((*sgn)%2 == 0 ? 1 : -1);
      }
    }
    
    template <typename REAL>
    __global__ void kernel_det_mod(const len_t m, const len_t n, const REAL *x, REAL *mod, int *sgn)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n && i == j)
      {
        REAL d = x[i + m*j];
        int s = 0;
        
        if (d < 0)
        {
          d = (*mod) + log(-d);
          s++;
        }
        else
          d = (*mod) + log(d);
        
        atomicAdd(mod, d);
        int s_g = (*sgn);
        (*sgn) = 0;
        atomicAdd(sgn, s);
        
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
          (*sgn) = ((*sgn)%2 == 0 ? 1 : -1);
          (*sgn) *= s_g;
        }
      }
    }
  }
  
  /**
    @brief Computes the determinant in logarithmic form.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] sign The sign of the determinant.
    @param[out] modulus Log of the modulus.
    
    @impl Uses `lu()`.
    
    @allocs Allocates temporary storage to compute the LU.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void det(gpumat<REAL> &x, int &sign, REAL &modulus)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    auto c = x.get_card();
    
    gpuvec<int> p(c);
    int info;
    lu(x, p, info);
    
    if (info != 0)
    {
      if (info > 0)
      {
        sign = 1;
        modulus = -INFINITY;
        return;
      }
      else
        return;
    }
    
    // get determinant
    modulus = 0.0;
    sign = 1;
    
    gpuscalar<int> sign_gpu(c, sign);
    gpuscalar<REAL> modulus_gpu(c, modulus);
    
    kernel_lu_pivot_sgn<<<p.get_griddim(), p.get_blockdim()>>>(p.size(),
      p.data_ptr(), sign_gpu.data_ptr());
    kernel_det_mod<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(), x.ncols(),
      x.data_ptr(), modulus_gpu.data_ptr(), sign_gpu.data_ptr());
    
    sign_gpu.get_val(&sign);
    modulus_gpu.get_val(&modulus);
  }
  
  
  
  /**
    @brief Computes the trace, i.e. the sum of the diagonal.
    
    @param[in] x Input data matrix.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  REAL trace(const gpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    auto c = x.get_card();
    
    REAL tr = 0;
    gpuscalar<REAL> tr_gpu(c, tr);
    
    fml::kernelfuns::kernel_trace<<<x.get_griddim(), x.get_blockdim()>>>(m, n,
      x.data_ptr(), tr_gpu.data_ptr());
    
    tr_gpu.get_val(&tr);
    c->check();
    
    return tr;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int svd_internals(const int nu, const int nv, gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
    {
      auto c = x.get_card();
      
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      s.resize(minmn);
      
      signed char jobu, jobvt;
      if (nu == 0 && nv == 0)
      {
        jobu = 'N';
        jobvt = 'N';
      }
      else //if (nu <= minmn && nv <= minmn)
      {
        jobu = 'S';
        jobvt = 'S';
        
        u.resize(m, minmn);
        vt.resize(minmn, n);
      }
      
      int lwork;
      gpulapack_status_t check = gpulapack::gesvd_buflen(c->lapack_handle(), m, n,
        x.data_ptr(), &lwork);
      gpulapack::err::check_ret(check, "gesvd_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      gpuvec<REAL> rwork(c, minmn-1);
      
      int info = 0;
      gpuscalar<int> info_device(c, info);
      
      check = gpulapack::gesvd(c->lapack_handle(), jobu, jobvt, m, n, x.data_ptr(),
        m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), minmn, work.data_ptr(),
        lwork, rwork.data_ptr(), info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "gesvd");
      
      return info;
    }
  }
  
  /**
    @brief Computes the singular value decomposition.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singnular vectors.
    
    @impl Uses the cuSOLVER function `cusolverDnXgesvd()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void svd(gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    err::check_card(x, s);
    
    gpumat<REAL> ignored(x.get_card());
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    fml::linalgutils::check_info(info, "gesvd");
  }
  
  /// \overload
  template <typename REAL>
  void svd(gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
  {
    err::check_card(x, s);
    err::check_card(x, u);
    err::check_card(x, vt);
    
    int info = svd_internals(1, 1, x, s, u, vt);
    fml::linalgutils::check_info(info, "gesvd");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, gpumat<REAL> &x,
      gpuvec<REAL> &values, gpumat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      auto c = x.get_card();
      
      len_t n = x.nrows();
      values.resize(n);
      
      cusolverEigMode_t jobz;
      if (only_values)
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;
      else
        jobz = CUSOLVER_EIG_MODE_VECTOR;
      
      int lwork;
      gpulapack_status_t check = gpulapack::syevd_buflen(c->lapack_handle(), jobz,
        GPUBLAS_FILL_L, n, x.data_ptr(), n, values.data_ptr(), &lwork);
      gpulapack::err::check_ret(check, "syevd_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      
      int info = 0;
      gpuscalar<int> info_device(c, info);
      
      check = gpulapack::syevd(c->lapack_handle(), jobz, GPUBLAS_FILL_L,
        n, x.data_ptr(), n, values.data_ptr(), work.data_ptr(), lwork,
        info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "syevd");
      fml::linalgutils::check_info(info, "syevd");
      
      if (!only_values)
      {
        vectors.resize(n, n);
        gpuhelpers::gpu2gpu(x, vectors);
      }
      
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
    
    @impl Uses the cuSOLVER functions `cusolverDnXsyevd()`.
    
    @allocs If any output's dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void eigen_sym(gpumat<REAL> &x, gpuvec<REAL> &values)
  {
    err::check_card(x, values);
    gpumat<REAL> ignored(x.get_card());
    
    int info = eig_sym_internals(true, x, values, ignored);
    fml::linalgutils::check_info(info, "syevd");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(gpumat<REAL> &x, gpuvec<REAL> &values, gpumat<REAL> &vectors)
  {
    err::check_card(x, values);
    err::check_card(x, vectors);
    
    int info = eig_sym_internals(false, x, values, vectors);
    fml::linalgutils::check_info(info, "syevd");
  }
  
  
  
  /**
    @brief Compute the matrix inverse.
    
    @details The input is replaced by its inverse, computed via a PLU.
    
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the cuSOLVER functions `cusolverDnXgetrf()` (LU) and
    `cusolverDnXgetrs()` (solve).
    
    @allocs LU pivot data is allocated internally. The inverse is computed in
    a copy before copying back to the input.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void invert(gpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    auto c = x.get_card();
    gpuvec<int> p(c);
    int info;
    lu(x, p, info);
    fml::linalgutils::check_info(info, "getrf");
    
    // Invert
    const len_t n = x.nrows();
    const len_t nrhs = n;
    gpumat<REAL> inv(c, n, nrhs);
    inv.fill_eye();
    
    gpuscalar<int> info_device(c, info);
    
    gpulapack_status_t check = gpulapack::getrs(c->lapack_handle(), GPUBLAS_OP_N, n,
      nrhs, x.data_ptr(), n, p.data_ptr(), inv.data_ptr(), n, info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "getrs");
    fml::linalgutils::check_info(info, "getrs");
    
    gpuhelpers::gpu2gpu(inv, x);
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void solver(gpumat<REAL> &x, len_t ylen, len_t nrhs, REAL *y_d)
    {
      const len_t n = x.nrows();
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      if (n != ylen)
        throw std::runtime_error("rhs 'y' must be compatible with data matrix 'x'");
      
      // Factor x = LU
      auto c = x.get_card();
      gpuvec<int> p(c);
      int info;
      lu(x, p, info);
      fml::linalgutils::check_info(info, "getrf");
      
      // Solve xb = y
      gpuscalar<int> info_device(c, info);
      
      gpulapack_status_t check = gpulapack::getrs(c->lapack_handle(), GPUBLAS_OP_N,
        n, nrhs, x.data_ptr(), n, p.data_ptr(), y_d, n, info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "getrs");
      fml::linalgutils::check_info(info, "getrs");
    }
  }
  
  /**
    @brief Solve a system of equations.
    
    @details The input is replaced by its PLU factorization.
    
    @param[inout] x Input LHS. Should be square. Overwritten by LU.
    @param[inout] y Input RHS. Overwritten by solution.
    
    @impl Uses the cuSOLVER functions `cusolverDnXgetrf()` (LU) and
    `cusolverDnXgetrs()` (solve).
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square or if the RHS is incompatible with the
    LHS, a `runtime_error` exception is thrown. If an allocation fails, a
    `bad_alloc` exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void solve(gpumat<REAL> &x, gpuvec<REAL> &y)
  {
    err::check_card(x, y);
    solver(x, y.size(), 1, y.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  void solve(gpumat<REAL> &x, gpumat<REAL> &y)
  {
    err::check_card(x, y);
    solver(x, y.nrows(), y.ncols(), y.data_ptr());
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void qr_internals(const bool pivot, gpumat<REAL> &x, gpuvec<REAL> &qraux, gpuvec<REAL> &work)
    {
      if (pivot)
        throw std::runtime_error("pivoting not supported at this time");
      
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      auto c = x.get_card();
      
      qraux.resize(minmn);
      
      int lwork;
      gpulapack_status_t check = gpulapack::geqrf_buflen(c->lapack_handle(), m,
        n, x.data_ptr(), m, &lwork);
      gpulapack::err::check_ret(check, "geqrf_bufferSize");
      
      if (lwork > work.size())
        work.resize(lwork);
      
      int info = 0;
      gpuscalar<int> info_device(c, info);
      
      check = gpulapack::geqrf(c->lapack_handle(), m, n, x.data_ptr(), m,
        qraux.data_ptr(), work.data_ptr(), lwork, info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "syevd");
      fml::linalgutils::check_info(info, "geqrf");
    }
  }
  
  /**
    @brief Computes the QR decomposition.
    
    @details The factorization works mostly in-place by modifying the input
    data. After execution, the matrix will be the LAPACK-like compact QR
    representation.
    
    @param[in] pivot NOTE Pivoting does not yet work on GPU. Should the factorization use column pivoting?
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] qraux Auxiliary data for compact QR.
    
    @impl Uses the cuSOLVER function `cusolverDnXgeqrf()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr(const bool pivot, gpumat<REAL> &x, gpuvec<REAL> &qraux)
  {
    err::check_card(x, qraux);
    gpuvec<REAL> work(x.get_card());
    qr_internals(pivot, x, qraux, work);
  }
  
  /**
    @brief Recover the Q matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[in] qraux Auxiliary data for compact QR.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl Uses the cuSOLVER function `cusolverDnXormqr()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_Q(const gpumat<REAL> &QR, const gpuvec<REAL> &qraux, gpumat<REAL> &Q,
    gpuvec<REAL> &work)
  {
    err::check_card(QR, qraux);
    err::check_card(QR, Q);
    err::check_card(QR, work);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    auto c = QR.get_card();
    
    int lwork;
    gpulapack_status_t check = gpulapack::ormqr_buflen(c->lapack_handle(),
      GPUBLAS_SIDE_LEFT, GPUBLAS_OP_N, m, minmn, minmn, QR.data_ptr(), m,
      qraux.data_ptr(), Q.data_ptr(), m, &lwork);
    
    if (lwork > work.size())
      work.resize(lwork);
    
    Q.resize(m, minmn);
    Q.fill_eye();
    
    int info = 0;
    gpuscalar<int> info_device(c, info);
    
    check = gpulapack::ormqr(c->lapack_handle(), GPUBLAS_SIDE_LEFT,
      GPUBLAS_OP_N, m, minmn, minmn, QR.data_ptr(), m, qraux.data_ptr(),
      Q.data_ptr(), m, work.data_ptr(), lwork, info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "ormqr");
    fml::linalgutils::check_info(info, "ormqr");
  }
  
  /**
    @brief Recover the R matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[out] R The R matrix.
    
    @impl Uses a custom LAPACK-like `lacpy()` clone.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_R(const gpumat<REAL> &QR, gpumat<REAL> &R)
  {
    err::check_card(QR, R);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    R.resize(minmn, n);
    R.fill_zero();
    fml::gpu_utils::lacpy(GPUBLAS_FILL_U, m, n, QR.data_ptr(), m, R.data_ptr(), minmn);
  }
  
  
  
  /**
    @brief Computes the LQ decomposition.
    
    @details The factorization works mostly in-place by modifying the input
    data. After execution, the matrix will be the LAPACK-like compact LQ
    representation.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] lqaux Auxiliary data for compact LQ.
    
    @impl NOTE: not directly supported by cuSOLVER (vendor gpulapack + cuda
    backend). In that case, the matrix is transposed and a QR is performed.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq(gpumat<REAL> &x, gpuvec<REAL> &lqaux)
  {
    err::check_card(x, lqaux);
    
    gpumat<REAL> tx(x.get_card());
    xpose(x, tx);
    
    gpuvec<REAL> work(x.get_card());
    qr_internals(false, tx, lqaux, work);
    
    xpose(tx, x);
  }
  
  /**
    @brief Recover the L matrix from an LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[out] L The L matrix.
    
    @impl NOTE: not directly supported by cuSOLVER (vendor gpulapack + cuda
    backend). In that case, the matrix is transposed and a QR is performed.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_L(const gpumat<REAL> &LQ, gpumat<REAL> &L)
  {
    err::check_card(LQ, L);
    
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    L.resize(m, minmn);
    L.fill_zero();
    fml::gpu_utils::lacpy(GPUBLAS_FILL_L, m, n, LQ.data_ptr(), m, L.data_ptr(), m);
  }
  
  /**
    @brief Recover the Q matrix from an LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[in] lqaux Auxiliary data for compact LQ.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl NOTE: not directly supported by cuSOLVER (vendor gpulapack + cuda
    backend). In that case, the matrix is transposed and a QR is performed.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_Q(const gpumat<REAL> &LQ, const gpuvec<REAL> &lqaux, gpumat<REAL> &Q,
    gpuvec<REAL> &work)
  {
    err::check_card(LQ, lqaux);
    err::check_card(LQ, Q);
    err::check_card(LQ, work);
    
    gpumat<REAL> QR(LQ.get_card());
    xpose(LQ, QR);
    
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    auto c = QR.get_card();
    
    int lwork;
    gpulapack_status_t check = gpulapack::ormqr_buflen(c->lapack_handle(),
      GPUBLAS_SIDE_RIGHT, GPUBLAS_OP_T, minmn, n, n, QR.data_ptr(), QR.nrows(),
      lqaux.data_ptr(), Q.data_ptr(), minmn, &lwork);
    
    if (lwork > work.size())
      work.resize(lwork);
    
    Q.resize(minmn, n);
    Q.fill_eye();
    
    int info = 0;
    gpuscalar<int> info_device(c, info);
    
    check = gpulapack::ormqr(c->lapack_handle(), GPUBLAS_SIDE_RIGHT,
      GPUBLAS_OP_T, minmn, n, n, QR.data_ptr(), QR.nrows(), lqaux.data_ptr(),
      Q.data_ptr(), minmn, work.data_ptr(), lwork, info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "ormqr");
    fml::linalgutils::check_info(info, "ormqr");
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
  void tssvd(gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
  {
    err::check_card(x, s);
    err::check_card(x, u);
    err::check_card(x, vt);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    auto c = x.get_card();
    
    gpuvec<REAL> qraux(c);
    gpuvec<REAL> work(c);
    qr_internals(false, x, qraux, work);
    
    gpumat<REAL> R(c, n, n);
    qr_R(x, R);
    
    gpumat<REAL> u_R(c, n, n);
    int info = svd_internals(1, 1, R, s, u_R, vt);
    fml::linalgutils::check_info(info, "gesvd");
    
    u.resize(m, n);
    qr_Q(x, qraux, u, work);
    
    matmult(false, false, (REAL)1.0, u, u_R, x);
    gpuhelpers::gpu2gpu(x, u);
  }
  
  /// \overload
  template <typename REAL>
  void tssvd(gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    err::check_card(x, s);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    auto c = x.get_card();
    s.resize(n);
    
    gpuvec<REAL> qraux(c);
    gpuvec<REAL> work(c);
    qr_internals(false, x, qraux, work);
    
    fml::gpu_utils::tri2zero('L', false, n, n, x.data_ptr(), m);
    
    int lwork;
    gpulapack_status_t check = gpulapack::gesvd_buflen(c->lapack_handle(), n, n,
      x.data_ptr(), &lwork);
    gpulapack::err::check_ret(check, "gesvd_bufferSize");
    
    if (lwork > work.size())
      work.resize(lwork);
    if (m-1 > qraux.size())
      qraux.resize(m-1);
    
    int info = 0;
    gpuscalar<int> info_device(c, info);
    
    check = gpulapack::gesvd(c->lapack_handle(), 'N', 'N', n, n, x.data_ptr(),
      m, s.data_ptr(), NULL, m, NULL, 1, work.data_ptr(), lwork,
      qraux.data_ptr(), info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "gesvd");
    fml::linalgutils::check_info(info, "gesvd");
  }
  
  
  
  /**
    @brief Computes the singular value decomposition using the
    "crossproducts SVD". This method is not numerically stable.
    
    @details The operation works by computing the crossproducts matrix X^T * X
    or X * X^T (whichever is smaller) and then computing the eigenvalue
    decomposition. 
    
    @param[inout] x Input data matrix.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `crossprod()` or `tcrossprod()` (whichever is smaller), and
    `eigen_sym()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void cpsvd(const gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
  {
    err::check_card(x, s);
    err::check_card(x, u);
    err::check_card(x, vt);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const len_t minmn = std::min(m, n);
    
    auto c = x.get_card();
    
    gpumat<REAL> cp(c);
    
    if (m >= n)
    {
      crossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, vt);
      vt.rev_cols();
      gpuhelpers::gpu2gpu(vt, cp);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, u);
      u.rev_cols();
      gpuhelpers::gpu2gpu(u, cp);
    }
    
    s.rev();
    auto sgrid = s.get_griddim();
    auto sblock = s.get_blockdim();
    fml::kernelfuns::kernel_root_abs<<<sgrid, sblock>>>(s.size(), s.data_ptr());
    
    REAL *ev_d;
    if (m >= n)
      ev_d = vt.data_ptr();
    else
      ev_d = cp.data_ptr();
    
    auto xgrid = x.get_griddim();
    auto xblock = x.get_blockdim();
    fml::kernelfuns::kernel_sweep_cols_div<<<xgrid, xblock>>>(minmn, minmn,
      ev_d, s.data_ptr());
    
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
  void cpsvd(const gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    err::check_card(x, s);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    auto c = x.get_card();
    
    gpumat<REAL> cp(c);
    
    if (m >= n)
      crossprod((REAL)1.0, x, cp);
    else
      tcrossprod((REAL)1.0, x, cp);
    
    eigen_sym(cp, s);
    
    s.rev();
    fml::kernelfuns::kernel_root_abs<<<s.get_griddim(), s.get_blockdim()>>>(s.size(), s.data_ptr());
  }
  
  
  
  /**
    @brief Compute the Choleski factorization.
    
    @details The matrix should be 1. square, 2. symmetric, 3. positive-definite.
    Failure of any of these conditions can lead to a runtime exception. The
    input is replaced by its lower-triangular Choleski factor.
    
    @param[inout] x Input data matrix, replaced by its lower-triangular Choleski
    factor.
    
    Uses the cuSOLVER function `cusolverDnXpotrf()`.
    
    @allocs Some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void chol(gpumat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (n != x.ncols())
      throw std::runtime_error("'x' must be a square matrix");
    
    auto c = x.get_card();
    const auto fill = GPUBLAS_FILL_L;
    
    int lwork;
    gpulapack_status_t check = gpulapack::potrf_buflen(c->lapack_handle(), fill, n,
      x.data_ptr(), n, &lwork);
    gpulapack::err::check_ret(check, "potrf_bufferSize");
    
    gpuvec<REAL> work(c, lwork);
    
    int info = 0;
    gpuscalar<int> info_device(c, info);
    check = gpulapack::potrf(c->lapack_handle(), fill, n, x.data_ptr(), n,
      work.data_ptr(), lwork, info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "potrf");
    if (info < 0)
      fml::linalgutils::check_info(info, "potrf");
    else if (info > 0)
      throw std::runtime_error("chol: leading minor of order " + std::to_string(info) + " is not positive definite");
    
    fml::gpu_utils::tri2zero('U', false, n, n, x.data_ptr(), n);
  }
}


#endif
