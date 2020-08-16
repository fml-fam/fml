// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_QR_H
#define FML_GPU_LINALG_LINALG_QR_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../internals/gpu_utils.hh"
#include "../internals/gpuscalar.hh"

#include "../gpumat.hh"
#include "../gpuvec.hh"

#include "linalg_err.hh"
#include "linalg_blas.hh"


namespace fml
{
namespace linalg
{
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
    gpulapack_status_t check = gpulapack::orgqr_buflen(c->lapack_handle(),
      m, minmn, minmn, QR.data_ptr(), m, qraux.data_ptr(), &lwork);
    
    if (lwork > work.size())
      work.resize(lwork);
    
    Q.resize(m, minmn);
    
    int info = 0;
    gpuscalar<int> info_device(c, info);
    
    
    fml::gpu_utils::lacpy('A', m, minmn, QR.data_ptr(), m, Q.data_ptr(), m);
    
    check = gpulapack::orgqr(c->lapack_handle(), m, minmn, minmn, Q.data_ptr(),
      m, qraux.data_ptr(), work.data_ptr(), lwork, info_device.data_ptr());
    
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
    fml::gpu_utils::lacpy('U', m, n, QR.data_ptr(), m, R.data_ptr(), minmn);
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
    fml::gpu_utils::lacpy('L', m, n, LQ.data_ptr(), m, L.data_ptr(), m);
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
}
}


#endif
