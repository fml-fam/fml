// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_QR_H
#define FML_MPI_LINALG_LINALG_QR_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../copy.hh"
#include "../mpimat.hh"

#include "linalg_blas.hh"
#include "linalg_err.hh"
#include "scalapack.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    template <typename REAL>
    void qr_internals(const bool pivot, mpimat<REAL> &x, cpuvec<REAL> &qraux, cpuvec<REAL> &work)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      const int *descx = x.desc_ptr();
      
      int info = 0;
      qraux.resize(minmn);
      
      REAL tmp;
      if (pivot)
        fml::scalapack::geqpf(m, n, NULL, descx, NULL, NULL, &tmp, -1, &info);
      else
        fml::scalapack::geqrf(m, n, NULL, descx, NULL, &tmp, -1, &info);
      
      int lwork = std::max((int) tmp, 1);
      if (lwork > work.size())
        work.resize(lwork);
      
      if (pivot)
      {
        cpuvec<int> p(n);
        p.fill_zero();
        fml::scalapack::geqpf(m, n, x.data_ptr(), descx, p.data_ptr(),
          qraux.data_ptr(), work.data_ptr(), lwork, &info);
      }
      else
        fml::scalapack::geqrf(m, n, x.data_ptr(), descx, qraux.data_ptr(),
          work.data_ptr(), lwork, &info);
      
      if (info != 0)
      {
        if (pivot)
          fml::linalgutils::check_info(info, "geqpf");
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
    
    @impl Uses the ScaLAPACK function `pXgeqpf()` if pivoting and `pXgeqrf()`
    otherwise.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr(const bool pivot, mpimat<REAL> &x, cpuvec<REAL> &qraux)
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
    
    @impl Uses the ScaLAPACK function `pXormgr()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_Q(const mpimat<REAL> &QR, const cpuvec<REAL> &qraux, mpimat<REAL> &Q, cpuvec<REAL> &work)
  {
    err::check_grid(QR, Q);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    const int *descQR = QR.desc_ptr();
    
    Q.resize(m, minmn);
    const int *descQ = Q.desc_ptr();
    
    int info = 0;
    REAL tmp;
    fml::scalapack::orgqr(m, minmn, minmn, NULL, descQR, NULL,
      &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::lacpy('A', m, minmn, QR.data_ptr(), descQR, Q.data_ptr(),
      descQ);
    
    fml::scalapack::orgqr(m, minmn, minmn, Q.data_ptr(), descQR,
      qraux.data_ptr(), work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "orgqr");
  }
  
  /**
    @brief Recover the R matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[out] R The R matrix.
    
    @impl Uses the ScaLAPACK function `pXlacpy()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_R(const mpimat<REAL> &QR, mpimat<REAL> &R)
  {
    err::check_grid(QR, R);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    R.resize(minmn, n);
    R.fill_zero();
    fml::scalapack::lacpy('U', m, n, QR.data_ptr(), QR.desc_ptr(), R.data_ptr(),
      R.desc_ptr());
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void lq_internals(mpimat<REAL> &x, cpuvec<REAL> &lqaux, cpuvec<REAL> &work)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      const int *descx = x.desc_ptr();
      
      int info = 0;
      lqaux.resize(minmn);
      
      REAL tmp;
      fml::scalapack::gelqf(m, n, NULL, descx, NULL, &tmp, -1, &info);
      int lwork = std::max((int) tmp, 1);
      if (lwork > work.size())
        work.resize(lwork);
      
      fml::scalapack::gelqf(m, n, x.data_ptr(), descx, lqaux.data_ptr(),
        work.data_ptr(), lwork, &info);
      
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
    
    @impl Uses the ScaLAPACK function `pXgelqf()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq(mpimat<REAL> &x, cpuvec<REAL> &lqaux)
  {
    cpuvec<REAL> work;
    lq_internals(x, lqaux, work);
  }
  
  /**
    @brief Recover the L matrix from a LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[out] L The L matrix.
    
    @impl Uses the ScaLAPACK function `pXlacpy()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_L(const mpimat<REAL> &LQ, mpimat<REAL> &L)
  {
    err::check_grid(LQ, L);
    
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    L.resize(m, minmn);
    L.fill_zero();
    
    fml::scalapack::lacpy('L', m, n, LQ.data_ptr(), LQ.desc_ptr(), L.data_ptr(),
      L.desc_ptr());
  }
  
  /**
    @brief Recover the Q matrix from a LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[in] lqaux Auxiliary data for compact LQ.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl Uses the ScaLAPACK function `pXorglq()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_Q(const mpimat<REAL> &LQ, const cpuvec<REAL> &lqaux, mpimat<REAL> &Q, cpuvec<REAL> &work)
  {
    err::check_grid(LQ, Q);
    
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    const int *descLQ = LQ.desc_ptr();
    
    Q.resize(minmn, n);
    const int *descQ = Q.desc_ptr();
    
    int info = 0;
    REAL tmp;
    fml::scalapack::orglq(minmn, n, minmn, NULL, descLQ, NULL,
      &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::lacpy('A', minmn, n, LQ.data_ptr(), descLQ, Q.data_ptr(),
      descQ);
    
    fml::scalapack::orglq(minmn, n, minmn, Q.data_ptr(), descQ,
      lqaux.data_ptr(), work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "orglq");
  }
}
}


#endif
