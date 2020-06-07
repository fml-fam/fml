// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_SVD_H
#define FML_CPU_LINALG_LINALG_SVD_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../internals/cpu_utils.hh"

#include "../copy.hh"
#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "linalg_factorizations.hh"


namespace fml
{
/// @brief Linear algebra functions.
namespace linalg
{
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
    copy::cpu2cpu(x, u);
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
    
    fml::cpu_utils::tri2zero('L', false, n, n, x.data_ptr(), m);
    
    int info = 0;
    cpuvec<int> iwork(8*n);
    
    REAL tmp;
    fml::lapack::gesdd('N', n, n, x.data_ptr(), m, s.data_ptr(), NULL, m, NULL,
      1, &tmp, -1, iwork.data_ptr(), &info);
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::lapack::gesdd('N', n, n, x.data_ptr(), m, s.data_ptr(), NULL, m, NULL,
      1, work.data_ptr(), lwork, iwork.data_ptr(), &info);
    fml::linalgutils::check_info(info, "gesdd");
  }
  
  
  
  /**
    @brief Computes the singular value decomposition for short/fat data.
    The number of columns must be greater than the number of rows. If the number
    of columns is not significantly larger than the number of rows, this may not
    be more efficient than simply calling `linalg::svd()`.
    
    @details The operation works by computing an LQ and then taking the SVD of
    the L matrix. The left singular vectors are Q times the right singular
    vectors from L's SVD, and the singular value and the left singular vectors
    are those from L's SVD.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `linalg::lq()` and `linalg::svd()`, and if computing the
    left/right singular vectors, `linalg::lq_L()` and `linalg::lq_Q()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void sfsvd(cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m >= n)
      throw std::runtime_error("'x' must have more cols than rows");
    
    cpuvec<REAL> lqaux;
    cpuvec<REAL> work;
    lq_internals(x, lqaux, work);
    
    cpumat<REAL> L(m, m);
    lq_L(x, L);
    
    cpumat<REAL> vt_L(m, m);
    svd(L, s, u, vt_L);
    
    vt.resize(n, m);
    vt.fill_eye();
    
    lq_Q(x, lqaux, vt, work);
    
    matmult(false, false, (REAL)1.0, vt_L, vt, x);
    copy::cpu2cpu(x, vt);
  }
  
  /// \overload
  template <typename REAL>
  void sfsvd(cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m >= n)
      throw std::runtime_error("'x' must have more cols than rows");
    
    s.resize(m);
    
    cpuvec<REAL> lqaux;
    cpuvec<REAL> work;
    lq_internals(x, lqaux, work);
    
    fml::cpu_utils::tri2zero('U', false, m, m, x.data_ptr(), m);
    
    int info = 0;
    cpuvec<int> iwork(8*n);
    
    REAL tmp;
    fml::lapack::gesdd('N', m, m, x.data_ptr(), m, s.data_ptr(), NULL, m, NULL,
      1, &tmp, -1, iwork.data_ptr(), &info);
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::lapack::gesdd('N', m, m, x.data_ptr(), m, s.data_ptr(), NULL, m, NULL,
      1, work.data_ptr(), lwork, iwork.data_ptr(), &info);
    fml::linalgutils::check_info(info, "gesdd");
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
  void cpsvd(const cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u,
    cpumat<REAL> &vt)
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
      copy::cpu2cpu(vt, cp);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, u);
      u.rev_cols();
      copy::cpu2cpu(u, cp);
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
    
    cpumat<REAL> cp;
    
    if (m >= n)
      crossprod((REAL)1.0, x, cp);
    else
      tcrossprod((REAL)1.0, x, cp);
    
    eigen_sym(cp, s);
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
  }
}
}


#endif
