// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_SVD_H
#define FML_GPU_LINALG_SVD_H
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

#include "crossprod.hh"
#include "eigen.hh"
#include "matmult.hh"
#include "qr.hh"
#include "xpose.hh"


namespace fml
{
namespace linalg
{
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
    void tssvd(gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
    {
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
      copy::gpu2gpu(x, u);
    }
    
    template <typename REAL>
    void tssvd(gpumat<REAL> &x, gpuvec<REAL> &s)
    {
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
    
    
    
    template <typename REAL>
    void sfsvd(gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      if (m >= n)
        throw std::runtime_error("'x' must have more cols than rows");
      
      gpumat<REAL> tx = xpose(x);
      gpumat<REAL> v(x.get_card());
      tssvd(tx, s, v, u);
      xpose(v, vt);
    }
    
    template <typename REAL>
    void sfsvd(gpumat<REAL> &x, gpuvec<REAL> &s)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      if (m >= n)
        throw std::runtime_error("'x' must have more cols than rows");
      
      auto tx = xpose(x);
      tssvd(tx, s);
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
    
    @impl Uses `linalg::qr()` to reduce the matrix to a square matrix, and then
    calls `linalg::svd()`. If computing the vectors, `linalg::qr_Q()` and
    `linalg::qr_R()`. Since cuSOLVER only offers QR, if m<n we operate on a
    transpose.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qrsvd(gpumat<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
  {
    err::check_card(x, s, u, vt);
    
    if (x.is_square())
      svd(x, s, u, vt);
    else if (x.nrows() > x.ncols())
      tssvd(x, s, u, vt);
    else
      sfsvd(x, s, u, vt);
  }
  
  /// \overload
  template <typename REAL>
  void qrsvd(gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    err::check_card(x, s);
    
    if (x.is_square())
      svd(x, s);
    else if (x.nrows() > x.ncols())
      tssvd(x, s);
    else
      sfsvd(x, s);
  }
  
  
  
  namespace
  {
    template <typename REAL>
    __global__ void kernel_sweep_cols_div(const len_t m, const len_t n, REAL *data, const REAL *v)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
          data[i + m*j] /= v[j];
    }
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
    err::check_card(x, s, u, vt);
    
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
      copy::gpu2gpu(vt, cp);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, u);
      u.rev_cols();
      copy::gpu2gpu(u, cp);
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
    kernel_sweep_cols_div<<<xgrid, xblock>>>(minmn, minmn, ev_d, s.data_ptr());
    
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
  
  
  
  namespace
  {
    template <typename REAL>
    void rsvd_A(const uint32_t seed, const int k, const int q, gpumat<REAL> &x,
      gpumat<REAL> &QY)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      gpumat<REAL> omega(x.get_card(), n, 2*k);
      omega.fill_runif(seed);
      
      gpumat<REAL> Y(x.get_card(), m, 2*k);
      gpumat<REAL> Z(x.get_card(), n, 2*k);
      gpumat<REAL> QZ(x.get_card(), n, 2*k);
      
      gpuvec<REAL> qraux(x.get_card());
      gpuvec<REAL> work(x.get_card());
      
      gpumat<REAL> B(x.get_card(), 2*k, n);
      
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
  void rsvd(const uint32_t seed, const int k, const int q, gpumat<REAL> &x,
    gpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    gpumat<REAL> QY(x.get_card(), m, 2*k);
    gpumat<REAL> B(x.get_card(), 2*k, n);
    
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
  void rsvd(const uint32_t seed, const int k, const int q, gpumat<REAL> &x,
    gpuvec<REAL> &s, gpumat<REAL> &u, gpumat<REAL> &vt)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    gpumat<REAL> QY(x.get_card(), m, 2*k);
    gpumat<REAL> B(x.get_card(), 2*k, n);
    
    // Stage A
    rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(true, false, (REAL)1.0, QY, x, B);
    
    gpumat<REAL> uB(x.get_card());
    qrsvd(B, s, uB, vt);
    
    s.resize(k);
    
    matmult(false, false, (REAL)1.0, QY, uB, u);
    u.resize(u.nrows(), k);
  }
}
}


#endif
