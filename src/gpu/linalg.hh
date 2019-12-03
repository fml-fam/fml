// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_H
#define FML_GPU_LINALG_H
#pragma once


#include <stdexcept>
#include <string>

#include "../_internals/linalgutils.hh"

#include "internals/launcher.hh"

#include "gpumat.hh"
#include "gpuvec.hh"


namespace linalg
{
  namespace
  {
    inline std::string get_cublas_error_msg(cublasStatus_t check)
    {
      if (check == CUBLAS_STATUS_SUCCESS)
        return "";
      else if (check == CUBLAS_STATUS_NOT_INITIALIZED)
        return "cuBLAS not initialized";
      else if (check == CUBLAS_STATUS_ALLOC_FAILED)
        return "internal cuBLAS memory allocation failed";
      else if (check == CUBLAS_STATUS_INVALID_VALUE)
        return "unsupported parameter";
      else if (check == CUBLAS_STATUS_ARCH_MISMATCH)
        return "function requires feature missing from device architecture";
      else if (check == CUBLAS_STATUS_MAPPING_ERROR)
        return "access to GPU memory space failed";
      else if (check == CUBLAS_STATUS_EXECUTION_FAILED)
        return "GPU program failed to execute";
      else if (check == CUBLAS_STATUS_INTERNAL_ERROR)
        return "internal cuBLAS operation failed";
      else if (check == CUBLAS_STATUS_NOT_SUPPORTED)
        return "requested functionality is not supported";
      else if (check == CUBLAS_STATUS_LICENSE_ERROR)
        return "error with cuBLAS license check";
      else
        return "unknown cuBLAS error occurred";
    }
    
    inline void check_cublas_ret(cublasStatus_t check, std::string op)
    {
      if (check != CUBLAS_STATUS_SUCCESS)
      {
        std::string msg = "cuBLAS " + op + "() failed with error: " + get_cublas_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
    
    
    
    inline std::string get_cusolver_error_msg(cusolverStatus_t check)
    {
      if (check == CUSOLVER_STATUS_SUCCESS)
        return "";
      else if (check == CUSOLVER_STATUS_NOT_INITIALIZED)
        return "cuSOLVER not initialized";
      else if (check == CUSOLVER_STATUS_ALLOC_FAILED)
        return "internal cuSOLVER memory allocation failed";
      else if (check == CUSOLVER_STATUS_INVALID_VALUE)
        return "unsupported parameter";
      else if (check == CUSOLVER_STATUS_ARCH_MISMATCH)
        return "function requires feature missing from device architecture";
      else if (check == CUSOLVER_STATUS_EXECUTION_FAILED)
        return "GPU program failed to execute";
      else if (check == CUSOLVER_STATUS_INTERNAL_ERROR)
        return "internal cuSOLVER operation failed";
      else if (check == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
        return "matrix type not supported";
      else
        return "unknown cuSOLVER error occurred";
    }
    
    inline void check_cusolver_ret(cusolverStatus_t check, std::string op)
    {
      if (check != CUSOLVER_STATUS_SUCCESS)
      {
        std::string msg = "cuSOLVER " + op + "() failed with error: " + get_cusolver_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
    
    
    
    inline void check_info(const int info, std::string fun)
    {
      if (info != 0)
      {
        std::string msg = "CUDA function " + fun + "() returned info=" + std::to_string(info);
        throw std::runtime_error(msg);
      }
    }
    
    
    
    template <typename REAL, class ARR>
    void check_card(const gpumat<REAL> &a, const ARR &b)
    {
      if (a.get_card()->get_id() != b.get_card()->get_id())
        throw std::runtime_error("gpumat/gpuvex objects must be allocated on the same gpu");
    }
  }
  
  
  
  // ret = alpha*op(x) + beta*op(y)
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const gpumat<REAL> &x, const gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    check_card(x, y);
    check_card(x, ret);
    
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    auto c = x.get_card();
    cublasOperation_t cbtransx = transx ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cbtransy = transy ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    culapack::geam(c->cb_handle(), cbtransx, cbtransy, m, n, alpha, x.data_ptr(), x.nrows(), beta, y.data_ptr(), y.nrows(), ret.data_ptr(), m);
  }
  
  template <typename REAL>
  gpumat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const gpumat<REAL> &x, const gpumat<REAL> &y)
  {
    check_card(x, y);
    
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    auto c = x.get_card();
    gpumat<REAL> ret(c, m, n);
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
  
  
  
  /**
   * @brief Returns alpha*op(x)*op(y) where op(A) is A or A^T
   * 
   * @param[in] transx Should x^T be used?
   * @param[in] transy Should y^T be used?
   * @param[in] alpha Scalar.
   * @param[in] x Left multiplicand.
   * @param[in] y Right multiplicand.
   * 
   * @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception.
   * 
   * @impl Uses the cuBLAS function `cublasXgemm()`.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  gpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const gpumat<REAL> &x, const gpumat<REAL> &y)
  {
    check_card(x, y);
    
    int m, n, k;
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    auto c = x.get_card();
    gpumat<REAL> ret(c, m, n);
    
    cublasOperation_t cbtransx = transx ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cbtransy = transy ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasStatus_t check = culapack::gemm(c->cb_handle(), cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), x.nrows(), y.data_ptr(), y.nrows(), (REAL)0, ret.data_ptr(), m);
    check_cublas_ret(check, "gemm");
    
    return ret;
  }
  
  /**
   * @brief Computes ret = alpha*op(x)*op(y) where op(A) is A or A^T
   * 
   * @param[in] transx Should x^T be used?
   * @param[in] transy Should y^T be used?
   * @param[in] alpha Scalar.
   * @param[in] x Left multiplicand.
   * @param[in] y Right multiplicand.
   * @param[out] ret The product.
   * 
   * @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception.
   * 
   * @impl Uses the cuBLAS function `cublasXgemm()`.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const gpumat<REAL> &x, const gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    check_card(x, y);
    check_card(x, ret);
    
    int m, n, k;
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    cublasOperation_t cbtransx = transx ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cbtransy = transy ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasStatus_t check = culapack::gemm(x.get_card()->cb_handle(), cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), x.nrows(), y.data_ptr(), y.nrows(), (REAL)0, ret.data_ptr(), m);
    check_cublas_ret(check, "gemm");
  }
  
  
  
  /**
   * @brief Computes lower triangle of alpha*x^T*x
   * 
   * @param[in] alpha Scalar.
   * @param[in] x Input data matrix.
   * @param[out] ret The product.
   * 
   * @impl Uses the cuBLAS function `cublasXsyrk()`.
   * 
   * @allocs If the output dimension is inappropriately sized, it will
   * automatically be re-allocated.
   * 
   * @except If a reallocation is triggered and fails, a `bad_alloc` exception
   * will be thrown.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const gpumat<REAL> &x, gpumat<REAL> &ret)
  {
    check_card(x, ret);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    ret.fill_zero();
    
    auto cbh = x.get_card()->cb_handle();
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    
    cublasStatus_t check = culapack::syrk(cbh, uplo, trans, n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
    check_cublas_ret(check, "syrk");
  }
  
  /**
   * \overload
   */
  template <typename REAL>
  gpumat<REAL> crossprod(const REAL alpha, const gpumat<REAL> &x)
  {
    const len_t n = x.ncols();
    gpumat<REAL> ret(x.get_card(), n, n);
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
   * @brief Computes lower triangle of alpha*x*x^T
   * 
   * @param[in] alpha Scalar.
   * @param[in] x Input data matrix.
   * @param[out] ret The product.
   * 
   * @impl Uses the cuBLAS function `cublasXsyrk()`.
   * 
   * @allocs If the output dimension is inappropriately sized, it will
   * automatically be re-allocated.
   * 
   * @except If a reallocation is triggered and fails, a `bad_alloc` exception
   * will be thrown.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void tcrossprod(const REAL alpha, const gpumat<REAL> &x, gpumat<REAL> &ret)
  {
    check_card(x, ret);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != ret.nrows() || m != ret.ncols())
      ret.resize(m, m);
    
    ret.fill_zero();
    
    auto cbh = x.get_card()->cb_handle();
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    
    cublasStatus_t check = culapack::syrk(cbh, uplo, trans, m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
    check_cublas_ret(check, "syrk");
  }
  
  /**
   * \overload
   */
  template <typename REAL>
  gpumat<REAL> tcrossprod(const REAL alpha, const gpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    gpumat<REAL> ret(x.get_card(), m, m);
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
   * @brief Computes the transpose out-of-place (i.e. in a copy).
   * 
   * @param[in] x Input data matrix.
   * @param[out] tx The transpose.
   * 
   * @impl Uses the cuBLAS function `cublasXgeam()`.
   * 
   * @allocs If the output dimension is inappropriately sized, it will
   * automatically be re-allocated.
   * 
   * @except If a reallocation is triggered and fails, a `bad_alloc` exception
   * will be thrown.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void xpose(const gpumat<REAL> &x, gpumat<REAL> &tx)
  {
    check_card(x, tx);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      tx.resize(n, m);
    
    auto cbh = x.get_card()->cb_handle();
    
    cublasStatus_t check = culapack::geam(cbh, CUBLAS_OP_T, CUBLAS_OP_N, n, m, (REAL)1.0, x.data_ptr(), m, (REAL) 0.0, tx.data_ptr(), n, tx.data_ptr(), n);
    check_cublas_ret(check, "geam");
  }
  
  /**
   * \overload
   */
  template <typename REAL>
  gpumat<REAL> xpose(const gpumat<REAL> &x)
  {
    gpumat<REAL> tx(x.get_card(), x.ncols(), x.nrows());
    xpose(x, tx);
    return tx;
  }
  
  
  
  /**
   * @brief Computes the PLU factorization with partial pivoting.
   * 
   * @details The input is replaced by its LU factorization, with L
   * unit-diagonal.
   * 
   * @param[inout] x Input data matrix, replaced by its LU factorization.
   * @param[out] p Vector of pivots, representing the diagonal matrix P in the
   * PLU.
   * 
   * @impl Uses the cuSOLVER function `cusolverDnXgetrf()`.
   * 
   * @allocs If the pivot vector is inappropriately sized, it will automatically
   * be re-allocated.
   * 
   * @except If a reallocation is triggered and fails, a `bad_alloc` exception
   * will be thrown.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  int lu(gpumat<REAL> &x, gpuvec<int> &p)
  {
    check_card(x, p);
    
    int info = 0;
    const int m = x.nrows();
    auto c = x.get_card();
    
    const len_t lipiv = std::min(m, x.ncols());
    if (!p.get_card()->valid_card())
      p.set(c);
    
    p.resize(lipiv);
    
    int lwork;
    cusolverStatus_t check = culapack::getrf_buflen(c->cs_handle(), m, m, x.data_ptr(), m, &lwork);
    check_cusolver_ret(check, "getrf_bufferSize");
    
    gpuvec<REAL> work(c, lwork);
    int *info_device = (int*) c->mem_alloc(sizeof(*info_device));
    c->mem_cpu2gpu(info_device, &info, sizeof(info));
    
    check = culapack::getrf(c->cs_handle(), m, m, x.data_ptr(), m, work.data_ptr(), p.data_ptr(), info_device);
    
    c->mem_gpu2cpu(&info, info_device, sizeof(info));
    c->mem_free(info_device);
    
    check_cusolver_ret(check, "getrf");
    
    return info;
  }
  
  /**
   * \overload
   */
  template <typename REAL>
  int lu(gpumat<REAL> &x)
  {
    gpuvec<int> p(x.get_card());
    return lu(x, p);
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
      
      s.resize(c, minmn);
      
      signed char jobu, jobvt;
      if (nu == 0 && nv == 0)
      {
        jobu = 'N';
        jobvt = 'N';
      }
      else //if (nu <= minmn && nv <= minmn)
      {
        jobu = 'V';
        jobvt = 'V';
        
        u.resize(c, m, minmn);
        vt.resize(c, minmn, n);
      }
      
      int lwork;
      cusolverStatus_t check = culapack::gesvd_buflen(c->cs_handle(), m, n,
        x.data_ptr(), &lwork);
      check_cusolver_ret(check, "gesvd_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      gpuvec<REAL> rwork(c, minmn-1);
      
      int info = 0;
      int *info_device = (int*) c->mem_alloc(sizeof(*info_device));
      c->mem_cpu2gpu(info_device, &info, sizeof(info));
      
      check = culapack::gesvd(c->cs_handle(), jobu, jobvt, m, n, x.data_ptr(),
        m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), minmn, work.data_ptr(),
        lwork, rwork.data_ptr(), info_device);
      
      c->mem_gpu2cpu(&info, info_device, sizeof(info));
      c->mem_free(info_device);
      
      check_cusolver_ret(check, "gesvd");
      
      return info;
    }
  }
  
  /**
   * @brief Computes the singular value decomposition.
   * 
   * @param[inout] x Input data matrix. Values are overwritten.
   * @param[out] s Vector of singular values.
   * @param[out] u Matrix of left singular vectors.
   * @param[out] vt Matrix of (transposed) right singnular vectors.
   * 
   * @impl Uses the cuSOLVER function `cusolverDnXgesvd()`.
   * 
   * @allocs If the any outputs are inappropriately sized, they will
   * automatically be re-allocated. Additionally, some temporary work storage
   * is needed.
   * 
   * @except If a (re-)allocation is triggered and fails, a `bad_alloc`
   * exception will be thrown.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void svd(gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    check_card(x, s);
    
    gpumat<REAL> ignored(x.get_card());
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    check_info(info, "gesvd");
  }
  
  /**
   * \overload
   */
  template <typename REAL>
  void svd(gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
  {
    check_card(x, s);
    check_card(x, u);
    check_card(x, vt);
    
    int info = svd_internals(1, 1, x, s, u, vt);
    check_info(info, "gesvd");
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
      values.resize(c, n);
      
      cusolverEigMode_t jobz;
      if (only_values)
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;
      else
        jobz = CUSOLVER_EIG_MODE_VECTOR;
      
      int lwork;
      cusolverStatus_t check = culapack::syevd_buflen(c->cs_handle(), jobz,
        CUBLAS_FILL_MODE_UPPER, n, x.data_ptr(), n, values.data_ptr(), &lwork);
      check_cusolver_ret(check, "syevd_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      
      int info = 0;
      int *info_device = (int*) c->mem_alloc(sizeof(*info_device));
      c->mem_cpu2gpu(info_device, &info, sizeof(info));
      
      check = culapack::syevd(c->cs_handle(), jobz, CUBLAS_FILL_MODE_UPPER, n,
        x.data_ptr(), n, values.data_ptr(), work.data_ptr(), lwork,
        info_device);
      
      c->mem_gpu2cpu(&info, info_device, sizeof(info));
      c->mem_free(info_device);
      
      check_cusolver_ret(check, "syevd");
      
      if (!only_values)
      {
        vectors.resize(c, n, n);
        gpuhelpers::gpu2gpu(x, vectors);
      }
      
      return info;
    }
  }
  
  template <typename REAL>
  void eigen(bool symmetric, gpumat<REAL> &x, gpuvec<REAL> &values)
  {
    check_card(x, values);
    
    gpumat<REAL> ignored(x.get_card());
    if (symmetric)
    {
      int info = eig_sym_internals(true, x, values, ignored);
      check_info(info, "syevd");
    }
    else
    {
      // TODO
    }
  }
  
  template <typename REAL>
  void eigen(bool symmetric, gpumat<REAL> &x, gpuvec<REAL> &values,
    gpumat<REAL> &vectors)
  {
    check_card(x, values);
    check_card(x, vectors);
    
    if (symmetric)
    {
      int info = eig_sym_internals(false, x, values, vectors);
      check_info(info, "syevd");
    }
    else
    {
      // TODO
    }
  }
  
  
  
  
  /**
   * @brief Compute the matrix inverse.
   * 
   * @details The input is replaced by its inverse, computed via a PLU.
   * 
   * @param[inout] x Input data matrix. Should be square.
   * 
   * @impl Uses the cuSOLVER functions `cusolverDnXgetrf()` (LU) and
   * `cusolverDnXgetrs()` (inverse).
   * 
   * @allocs LU pivot data is allocated internally.
   * 
   * @except If the matrix is non-square, a `runtime_error` exception is thrown.
   * If an allocation fails, a `bad_alloc` exception will be thrown.
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void invert(gpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    auto c = x.get_card();
    gpuvec<int> p(c);
    int info = lu(x, p);
    check_info(info, "getrf");
    
    // Invert
    const len_t n = x.nrows();
    const len_t nrhs = n;
    gpumat<REAL> inv(c, n, nrhs);
    inv.fill_eye();
    
    int *info_device = (int*) c->mem_alloc(sizeof(*info_device));
    c->mem_cpu2gpu(info_device, &info, sizeof(info));
    
    cusolverStatus_t check = culapack::getrs(c->cs_handle(), CUBLAS_OP_N, n, nrhs, x.data_ptr(), n, p.data_ptr(), inv.data_ptr(), n, info_device);
    c->mem_gpu2cpu(&info, info_device, sizeof(info));
    c->mem_free(info_device);
    
    check_cusolver_ret(check, "getrs");
    check_info(info, "getrs");
    
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
      int info = lu(x, p);
      check_info(info, "getrf");
      
      // Solve xb = y
      int *info_device = (int*) c->mem_alloc(sizeof(*info_device));
      c->mem_cpu2gpu(info_device, &info, sizeof(info));
      
      cusolverStatus_t check = culapack::getrs(c->cs_handle(), CUBLAS_OP_N, n, nrhs, x.data_ptr(), n, p.data_ptr(), y_d, n, info_device);
      c->mem_gpu2cpu(&info, info_device, sizeof(info));
      c->mem_free(info_device);
      
      check_cusolver_ret(check, "getrs");
      check_info(info, "getrs");
    }
  }
  
  template <typename REAL>
  void solve(gpumat<REAL> &x, gpuvec<REAL> &y)
  {
    check_card(x, y);
    solver(x, y.size(), 1, y.data_ptr());
  }
  
  template <typename REAL>
  void solve(gpumat<REAL> &x, gpumat<REAL> &y)
  {
    check_card(x, y);
    solver(x, y.nrows(), y.ncols(), y.data_ptr());
  }
}


#endif
