// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_QR_H
#define FML_CPU_LINALG_LINALG_QR_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../internals/cpu_utils.hh"

#include "../copy.hh"
#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "lapack.hh"


namespace fml
{
namespace linalg
{
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
  void qr_Q(const cpumat<REAL> &QR, const cpuvec<REAL> &qraux, cpumat<REAL> &Q,
    cpuvec<REAL> &work)
  {
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    int info = 0;
    REAL tmp;
    fml::lapack::orgqr(m, minmn, minmn, QR.data_ptr(), m, NULL,
      &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    Q.resize(m, minmn);
    fml::lapack::lacpy('A', m, minmn, QR.data_ptr(), m, Q.data_ptr(), m);
    
    fml::lapack::orgqr(m, minmn, minmn, Q.data_ptr(), m, qraux.data_ptr(),
      work.data_ptr(), lwork, &info);
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
    const len_t minmn = std::min(m, n);
    
    R.resize(minmn, n);
    R.fill_zero();
    fml::lapack::lacpy('U', m, n, QR.data_ptr(), m, R.data_ptr(), minmn);
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void lq_internals(cpumat<REAL> &x, cpuvec<REAL> &lqaux, cpuvec<REAL> &work)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      int info = 0;
      lqaux.resize(minmn);
      
      REAL tmp;
      fml::lapack::gelqf(m, n, NULL, m, NULL, &tmp, -1, &info);
      int lwork = std::max((int) tmp, 1);
      if (lwork > work.size())
        work.resize(lwork);
      
      fml::lapack::gelqf(m, n, x.data_ptr(), m, lqaux.data_ptr(), work.data_ptr(),
        lwork, &info);
      
      if (info != 0)
        fml::linalgutils::check_info(info, "gelqf");
    }
  }
  
  /**
    @brief Computes the LQ decomposition.
    
    @details The factorization works mostly in-place by modifying the input
    data. After execution, the matrix will be the LAPACK-like compact LQ
    representation.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] lqaux Auxiliary data for compact LQ.
    
    @impl Uses the LAPACK function `Xgelqf()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq(cpumat<REAL> &x, cpuvec<REAL> &lqaux)
  {
    cpuvec<REAL> work;
    lq_internals(x, lqaux, work);
  }
  
  /**
    @brief Recover the L matrix from an LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[out] L The L matrix.
    
    @impl Uses the LAPACK function `Xlacpy()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_L(const cpumat<REAL> &LQ, cpumat<REAL> &L)
  {
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    L.resize(m, minmn);
    L.fill_zero();
    
    fml::lapack::lacpy('L', m, n, LQ.data_ptr(), m, L.data_ptr(), m);
  }
  
  /**
    @brief Recover the Q matrix from an LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[in] lqaux Auxiliary data for compact LQ.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl Uses the LAPACK function `Xormlq()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_Q(const cpumat<REAL> &LQ, const cpuvec<REAL> &lqaux, cpumat<REAL> &Q,
    cpuvec<REAL> &work)
  {
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    int info = 0;
    REAL tmp;
    fml::lapack::ormlq('R', 'N', minmn, n, minmn, LQ.data_ptr(), m, NULL,
      NULL, minmn, &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    Q.resize(minmn, n);
    Q.fill_eye();
    
    fml::lapack::ormlq('R', 'N', minmn, n, minmn, LQ.data_ptr(), m, lqaux.data_ptr(),
      Q.data_ptr(), minmn, work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "ormlq");
  }
}
}


#endif
