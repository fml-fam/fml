// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_SVD_H
#define FML_GPU_LINALG_LINALG_SVD_H
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

#include "linalg_factorizations.hh"


namespace fml
{
namespace linalg
{
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
    err::check_card(x, s);
    err::check_card(x, u);
    err::check_card(x, vt);
    
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
}
}


#endif
