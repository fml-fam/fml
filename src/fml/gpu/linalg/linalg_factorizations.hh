// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_FACTORIZATIONS_H
#define FML_GPU_LINALG_LINALG_FACTORIZATIONS_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../internals/gpu_utils.hh"
#include "../internals/gpuscalar.hh"
#include "../internals/kernelfuns.hh"

#include "../copy.hh"
#include "../gpumat.hh"
#include "../gpuvec.hh"

#include "linalg_err.hh"
#include "linalg_blas.hh"


namespace fml
{
namespace linalg
{
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
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    auto c = x.get_card();
    
    const len_t lipiv = std::min(m, n);
    if (!p.get_card()->valid_card())
      p.inherit(c);
    
    p.resize(lipiv);
    
    #if defined(FML_GPULAPACK_VENDOR)
      int lwork;
      gpulapack_status_t check = gpulapack::getrf_buflen(c->lapack_handle(), m,
        n, x.data_ptr(), m, &lwork);
      gpulapack::err::check_ret(check, "getrf_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      gpuscalar<int> info_device(c, info);
      
      check = gpulapack::getrf(c->lapack_handle(), m, n, x.data_ptr(), m,
        work.data_ptr(), p.data_ptr(), info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "getrf");
    #elif defined(FML_GPULAPACK_MAGMA)
      cpuvec<int> p_cpu(lipiv);
      gpulapack::getrf(m, n, x.data_ptr(), m, p_cpu.data_ptr(), &info);
      copy::cpu2gpu(p_cpu, p);
    #else
      #error "Unsupported GPU lapack"
    #endif
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
    
    @impl Uses the cuSOLVER function `cusolverDnXgesvd()`. Since cuSOLVER only
    supports the m>=n case, if m<n we operate on a transpose.
    
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
    err::check_card(x, s, u, vt);
    
    if (x.nrows() >= x.ncols())
    {
      int info = svd_internals(1, 1, x, s, u, vt);
      fml::linalgutils::check_info(info, "gesvd");
    }
    else
    {
      auto tx = xpose(x);
      gpumat<REAL> v(x.get_card());
      int info = svd_internals(1, 1, tx, s, v, u);
      xpose(v, vt);
      fml::linalgutils::check_info(info, "gesvd");
    }
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
        copy::gpu2gpu(x, vectors);
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
    err::check_card(x, values, values);
    
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
    
    copy::gpu2gpu(inv, x);
  }
  
  
  
  /**
    @brief Compute the matrix inverse of a triangular matrix.
    
    @details The input is replaced by its inverse.
    
    @param[in] upper Should the upper triangle be used? Otherwise the lower
    triangle will be used.
    @param[in] unit_diag Is the input matrix unit diagonal?
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the cuBLAS functions `cublasXtrsm()`.
    
    @allocs The inverse is computed in a copy.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void trinv(const bool upper, const bool unit_diag, gpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    const len_t n = x.nrows();
    gpumat<REAL> inv(x.get_card(), n, n);
    inv.fill_eye();
    
    gpublas_fillmode_t uplo = (upper ? GPUBLAS_FILL_U : GPUBLAS_FILL_L);
    gpublas_diagtype_t diag = (unit_diag ? GPUBLAS_DIAG_UNIT : GPUBLAS_DIAG_NON_UNIT);
    
    gpublas_status_t check =  gpublas::trsm(x.get_card()->blas_handle(),
      GPUBLAS_SIDE_LEFT, uplo, GPUBLAS_OP_N, diag, n, n, x.data_ptr(), n,
      inv.data_ptr(), n);
      
    gpublas::err::check_ret(check, "trsm");
    copy::gpu2gpu(inv, x);
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
    err::check_card(QR, qraux, Q, work);
    
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
    err::check_card(LQ, lqaux, Q, work);
    
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
}


#endif
