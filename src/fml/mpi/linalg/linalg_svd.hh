// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_SVD_H
#define FML_MPI_LINALG_LINALG_SVD_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../copy.hh"
#include "../mpimat.hh"

#include "linalg_blas.hh"
#include "linalg_eigen.hh"
#include "linalg_qr.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    namespace
    {
      template <typename REAL>
      int svd_internals(const int nu, const int nv, mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
      {
        int info = 0;
        char jobu, jobvt;
        
        const len_t m = x.nrows();
        const len_t n = x.ncols();
        const len_t minmn = std::min(m, n);
        
        s.resize(minmn);
        
        if (nu == 0 && nv == 0)
        {
          jobu = 'N';
          jobvt = 'N';
        }
        else // if (nu <= minmn && nv <= minmn)
        {
          jobu = 'V';
          jobvt = 'V';
          
          const int mb = x.bf_rows();
          const int nb = x.bf_cols();
          
          u.resize(m, minmn, mb, nb);
          vt.resize(minmn, n, mb, nb);
        }
        
        REAL tmp;
        fml::scalapack::gesvd(jobu, jobvt, m, n, x.data_ptr(), x.desc_ptr(), s.data_ptr(), u.data_ptr(), u.desc_ptr(), vt.data_ptr(), vt.desc_ptr(), &tmp, -1, &info);
        int lwork = (int) tmp;
        cpuvec<REAL> work(lwork);
        
        fml::scalapack::gesvd(jobu, jobvt, m, n, x.data_ptr(), x.desc_ptr(), s.data_ptr(), u.data_ptr(), u.desc_ptr(), vt.data_ptr(), vt.desc_ptr(), work.data_ptr(), lwork, &info);
        
        return info;
      }
    }
    
    /**
      @brief Computes the singular value decomposition.
      
      @param[inout] x Input data matrix. Values are overwritten.
      @param[out] s Vector of singular values.
      @param[out] u Matrix of left singular vectors.
      @param[out] vt Matrix of (transposed) right singnular vectors.
      
      @impl Uses the ScaLAPACK function `pXgesvd()`.
      
      @allocs If the any outputs are inappropriately sized, they will
      automatically be re-allocated. Additionally, some temporary work storage
      is needed.
      
      @except If a (re-)allocation is triggered and fails, a `bad_alloc`
      exception will be thrown.
      
      @comm The method will communicate across all processes in the BLACS grid.
      
      @tparam REAL should be 'float' or 'double'.
     */
    template <typename REAL>
    void svd(mpimat<REAL> &x, cpuvec<REAL> &s)
    {
      mpimat<REAL> ignored(x.get_grid());
      int info = svd_internals(0, 0, x, s, ignored, ignored);
      fml::linalgutils::check_info(info, "gesvd");
    }
    
    /// \overload
    template <typename REAL>
    void svd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
    {
      err::check_grid(x, u, vt);
      
      int info = svd_internals(1, 1, x, s, u, vt);
      fml::linalgutils::check_info(info, "gesvd");
    }
    
    
    
    template <typename REAL>
    void tssvd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      if (m <= n)
        throw std::runtime_error("'x' must have more rows than cols");
      
      const grid g = x.get_grid();
      
      cpuvec<REAL> qraux(n);
      cpuvec<REAL> work(m);
      qr_internals(false, x, qraux, work);
      
      mpimat<REAL> R(g, n, n, x.bf_rows(), x.bf_cols());
      qr_R(x, R);
      
      mpimat<REAL> u_R(g, n, n, x.bf_rows(), x.bf_cols());
      svd(R, s, u_R, vt);
      
      u.resize(m, n);
      u.fill_eye();
      
      qr_Q(x, qraux, u, work);
      
      matmult(false, false, (REAL)1.0, u, u_R, x);
      copy::mpi2mpi(x, u);
    }
    
    template <typename REAL>
    void tssvd(mpimat<REAL> &x, cpuvec<REAL> &s)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      if (m <= n)
        throw std::runtime_error("'x' must have more rows than cols");
      
      const grid g = x.get_grid();
      s.resize(n);
      
      cpuvec<REAL> qraux(n);
      cpuvec<REAL> work(m);
      qr_internals(false, x, qraux, work);
      
      fml::mpi_utils::tri2zero('L', false, g, n, n, x.data_ptr(), x.desc_ptr());
      
      int info = 0;
      
      REAL tmp;
      fml::scalapack::gesvd('N', 'N', n, n, x.data_ptr(), x.desc_ptr(),
        s.data_ptr(), NULL, NULL, NULL, NULL, &tmp, -1, &info);
      int lwork = (int) tmp;
      if (lwork > work.size())
        work.resize(lwork);
      
      fml::scalapack::gesvd('N', 'N', n, n, x.data_ptr(), x.desc_ptr(),
        s.data_ptr(), NULL, NULL, NULL, NULL, work.data_ptr(), lwork, &info);
      fml::linalgutils::check_info(info, "gesvd");
    }
    
    
    
    template <typename REAL>
    void sfsvd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      if (m >= n)
        throw std::runtime_error("'x' must have more cols than rows");
      
      const grid g = x.get_grid();
      
      cpuvec<REAL> lqaux;
      cpuvec<REAL> work;
      lq_internals(x, lqaux, work);
      
      mpimat<REAL> L(g, m, m, x.bf_rows(), x.bf_cols());
      lq_L(x, L);
      
      mpimat<REAL> vt_L(g, m, m, x.bf_rows(), x.bf_cols());
      svd(L, s, u, vt_L);
      
      vt.resize(n, m);
      vt.fill_eye();
      
      lq_Q(x, lqaux, vt, work);
      
      matmult(false, false, (REAL)1.0, vt_L, vt, x);
      copy::mpi2mpi(x, vt);
    }
    
    template <typename REAL>
    void sfsvd(mpimat<REAL> &x, cpuvec<REAL> &s)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      if (m >= n)
        throw std::runtime_error("'x' must have more cols than rows");
      
      const grid g = x.get_grid();
      s.resize(m);
      
      cpuvec<REAL> lqaux;
      cpuvec<REAL> work;
      lq_internals(x, lqaux, work);
      
      fml::mpi_utils::tri2zero('U', false, g, m, m, x.data_ptr(), x.desc_ptr());
      
      int info = 0;
      
      REAL tmp;
      fml::scalapack::gesvd('N', 'N', m, m, x.data_ptr(), x.desc_ptr(),
        s.data_ptr(), NULL, NULL, NULL, NULL, &tmp, -1, &info);
      int lwork = (int) tmp;
      if (lwork > work.size())
        work.resize(lwork);
      
      fml::scalapack::gesvd('N', 'N', m, m, x.data_ptr(), x.desc_ptr(),
        s.data_ptr(), NULL, NULL, NULL, NULL, work.data_ptr(), lwork, &info);
      fml::linalgutils::check_info(info, "gesvd");
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
  void qrsvd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    err::check_grid(x, u, vt);
    
    if (x.is_square())
      svd(x, s, u, vt);
    else if (x.nrows() > x.ncols())
      tssvd(x, s, u, vt);
    else
      sfsvd(x, s, u, vt);
  }
  
  /// \overload
  template <typename REAL>
  void qrsvd(mpimat<REAL> &x, cpuvec<REAL> &s)
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
  void cpsvd(const mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    err::check_grid(x, u, vt);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const len_t minmn = std::min(m, n);
    
    const grid g = x.get_grid();
    mpimat<REAL> cp(g, x.bf_rows(), x.bf_cols());
    
    if (m >= n)
    {
      crossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, vt);
      vt.rev_cols();
      copy::mpi2mpi(vt, cp);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, u);
      u.rev_cols();
      copy::mpi2mpi(u, cp);
    }
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
    
    len_t m_local, n_local;
    REAL *ev_d;
    if (m >= n)
    {
      m_local = vt.nrows_local();
      n_local = vt.ncols_local();
      ev_d = vt.data_ptr();
    }
    else
    {
      m_local = cp.nrows_local();
      n_local = cp.ncols_local();
      ev_d = cp.data_ptr();
    }
    
    for (len_t j=0; j<n_local; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<m_local; i++)
      {
        const int gi = fml::bcutils::l2g(i, x.bf_rows(), g.nprow(), g.myrow());
        const int gj = fml::bcutils::l2g(j, x.bf_cols(), g.npcol(), g.mycol());
        
        if (gi < minmn && gj < minmn)
          ev_d[i + m_local*j] /= s_d[gj];
      }
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
  void cpsvd(const mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    const grid g = x.get_grid();
    mpimat<REAL> cp(g, x.bf_rows(), x.bf_cols());
    
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
    void rsvd_A(const uint32_t seed, const int k, const int q, mpimat<REAL> &x,
      mpimat<REAL> &QY)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      const int mb = x.bf_rows();
      const int nb = x.bf_cols();
      
      auto g = x.get_grid();
      
      mpimat<REAL> omega(g, n, 2*k, mb, nb);
      omega.fill_runif(seed);
      
      mpimat<REAL> Y(g, m, 2*k, mb, nb);
      mpimat<REAL> Z(g, n, 2*k, mb, nb);
      mpimat<REAL> QZ(g, n, 2*k, mb, nb);
      
      cpuvec<REAL> qraux;
      cpuvec<REAL> work;
      
      mpimat<REAL> B(g, 2*k, n, mb, nb);
      
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
  void rsvd(const uint32_t seed, const int k, const int q, mpimat<REAL> &x,
    cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    mpimat<REAL> QY(x.get_grid(), m, 2*k, x.bf_rows(), x.bf_cols());
    mpimat<REAL> B(x.get_grid(), 2*k, n, x.bf_rows(), x.bf_cols());
    
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
  void rsvd(const uint32_t seed, const int k, const int q, mpimat<REAL> &x,
    cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    err::check_grid(x, u, vt);
    
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
