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

#include "../copy.hh"
#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "linalg_blas.hh"
#include "linalg_eigen.hh"
#include "linalg_qr.hh"


namespace fml
{
namespace linalg
{
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
  }
  
  /**
    @brief Computes the singular value decomposition by first reducing the
    rectangular matrix to a square matrix using an orthogonal factorization. If
    the matrix is square, we skip the orthogonal factorization.
    
    @details If the matrix has more rows than columns, the operation works by
    computing a QR and then taking the SVD of the R matrix. The left singular
    vectors are Q times the left singular vectors from R's SVD, and the singular
    value and the right singular vectors are those from R's SVD. Likewise, if
    the matrix has more columns than rows, w take the LQ and then the SVD of
    the L matrix. The left singular vectors are Q times the right singular
    vectors from L's SVD, and the singular value and the left singular vectors
    are those from L's SVD.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `linalg::qr()` or `linalg::lq()` (whichever is cheaper) to reduce
    the matrix to a square matrix, and then calls `linalg::svd()`. If computing
    the vectors, `linalg::qr_Q()` and `linalg::qr_R()` or `linalg::lq_L()` and
    `linalg::lq_Q()` are called.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qrsvd(cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    if (x.is_square())
      svd(x, s, u, vt);
    else if (x.nrows() > x.ncols())
      tssvd(x, s, u, vt);
    else
      sfsvd(x, s, u, vt);
  }
  
  /// \overload
  template <typename REAL>
  void qrsvd(cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    if (x.is_square())
      svd(x, s);
    else if (x.nrows() > x.ncols())
      tssvd(x, s);
    else
      sfsvd(x, s);
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
    
    #pragma omp parallel for if(minmn*minmn > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<minmn; j++)
    {
      #pragma omp simd
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
  
  
  
  namespace
  {
    template <typename REAL>
    void rsvd_A(const uint32_t seed, const int k, const int q, cpumat<REAL> &x,
      cpumat<REAL> &QY)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      cpumat<REAL> omega(n, 2*k);
      omega.fill_runif(seed);
      
      cpumat<REAL> Y(m, 2*k);
      cpumat<REAL> Z(n, 2*k);
      cpumat<REAL> QZ(n, 2*k);
      
      cpuvec<REAL> qraux;
      cpuvec<REAL> work;
      
      cpumat<REAL> B(2*k, n);
      
      // Stage A
      matmult(false, false, (REAL)1.0, x, omega, Y);
      qr_internals(false, Y, qraux, work);
      qr_Q(Y, qraux, QY, work);
      
      for (int i=0; i<q; i++)
      {
        matmult(true, false, (REAL)1.0, x, QY, Z);
        qr_internals(false, Z, qraux, work);
        qr_Q(Z, qraux, QZ, work);
        
        matmult(false, false, (REAL)1.0, x, QZ, Y);
        qr_internals(false, Y, qraux, work);
        qr_Q(Y, qraux, QY, work);
      }
    }
  }
  
  /**
    @brief Computes the truncated singular value decomposition using the
    normal projections method of Halko et al. This method is only an
    approximation.
    
    @param[inout] x Input data matrix.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses a series of QR's and matrix multiplications.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rsvd(const uint32_t seed, const int k, const int q, cpumat<REAL> &x,
    cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    cpumat<REAL> QY(m, 2*k);
    cpumat<REAL> B(2*k, n);
    
    // Stage A
    rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(true, false, (REAL)1.0, QY, x, B);
    
    cpumat<REAL> uB;
    svd(B, s);
    
    s.resize(k);
  }
  
  /// \overload
  template <typename REAL>
  void rsvd(const uint32_t seed, const int k, const int q, cpumat<REAL> &x,
    cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    cpumat<REAL> QY(m, 2*k);
    cpumat<REAL> B(2*k, n);
    
    // Stage A
    rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(true, false, (REAL)1.0, QY, x, B);
    
    cpumat<REAL> uB;
    svd(B, s, uB, vt);
    
    s.resize(k);
    
    matmult(false, false, (REAL)1.0, QY, uB, u);
    u.resize(u.nrows(), k);
  }
}
}


#endif
