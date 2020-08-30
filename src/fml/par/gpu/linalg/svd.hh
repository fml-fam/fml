// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_LINALG_SVD_H
#define FML_PAR_GPU_LINALG_SVD_H
#pragma once


#include "../parmat.hh"

#include "blas.hh"
#include "qr.hh"

#include "../../../gpu/linalg/linalg_blas.hh"
#include "../../../gpu/linalg/linalg_invert.hh"
#include "../../../gpu/linalg/linalg_qr.hh"
#include "../../../gpu/linalg/linalg_svd.hh"

#include "../../../gpu/copy.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the singular value decomposition using the "crossproducts
    SVD". This method is not numerically stable.
    
    @details The operation works by computing the crossproducts matrix X^T * X
    and then computing the eigenvalue decomposition. 
    
    @param[inout] x Input data matrix.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses a crossproduct which requires communication, followed by a local
    `linalg::eigen_sym()` call.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void cpsvd(const parmat_gpu<REAL> &x, gpuvec<REAL> &s)
  {
    const len_t n = x.ncols();
    
    auto cp = crossprod((REAL)1.0, x.data_obj());
    x.get_comm().allreduce(n*n, cp.data_ptr());
    eigen_sym(cp, s);
  }
  
  /// \overload
  template <typename REAL>
  void cpsvd(const parmat_gpu<REAL> &x, gpuvec<REAL> &s, gpumat<REAL> &u,
    gpumat<REAL> &vt)
  {
    const len_t n = x.ncols();
    
    auto cp = crossprod((REAL)1.0, x.data_obj());
    x.get_comm().allreduce(n*n, cp.data_ptr());
    eigen_sym(cp, s, vt);
    
    auto c = vt.get_card();
    
    s.rev();
    REAL *s_d = s.data_ptr();
    auto sgrid = s.get_griddim();
    auto sblock = s.get_blockdim();
    kernelfuns::kernel_root_abs<<<sgrid, sblock>>>(s.size(), s_d);
      
    REAL *vt_d = vt.data_ptr();
    vt.rev_cols();
    copy::gpu2gpu(vt, cp);
    
    auto xgrid = vt.get_griddim();
    auto xblock = vt.get_blockdim();
    kernelfuns::kernel_sweep_cols_div<<<xgrid, xblock>>>(n, n, vt_d, s_d);
      
    matmult(false, false, (REAL)1.0, x.data_obj(), vt, u);
    xpose(cp, vt);
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
    
    @impl Uses the TSQR implementation to reduce the matrix to a square matrix,
    and then calls `linalg::svd()`. If computing the vectors, `linalg::trinv()`
    and a series of matrix multiplications are used.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void tssvd(parmat_gpu<REAL> &x, gpuvec<REAL> &s)
  {
    gpumat<REAL> R(s.get_card());
    qr_R(mpi::REDUCE_TO_ALL, x, R);
    svd(R, s);
  }
  
  /// \overload
  template <typename REAL>
  void tssvd(parmat_gpu<REAL> &x, gpuvec<REAL> &s,
    parmat_gpu<REAL> &u, gpumat<REAL> &vt)
  {
    const len_t n = x.ncols();
    if (x.nrows() < (len_global_t)n)
      throw std::runtime_error("impossible dimensions");
    
    gpumat<REAL> x_cpy = x.data_obj().dupe();
    
    gpumat<REAL> R_local(s.get_card());
    gpuvec<REAL> qraux(s.get_card());
    qr(false, x.data_obj(), qraux);
    qr_R(x.data_obj(), R_local);
    
    gpumat<REAL> R(s.get_card(), n, n);
    tsqr::qr_allreduce(mpi::REDUCE_TO_ALL, n, n, R_local.data_ptr(), R.data_ptr(), x.get_comm().get_comm());
    
    copy::gpu2gpu(R, R_local);
    gpumat<REAL> u_R(s.get_card());
    svd(R_local, s, u_R, vt);
    
    // u = Q*u_R = x*R^{-1}*u_R
    u.resize(x.nrows(), x.ncols());
    trinv(true, false, R);
    matmult(false, false, (REAL)1, x_cpy, R, x.data_obj());
    matmult(false, false, (REAL)1, x.data_obj(), u_R, u.data_obj());
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    void rsvd_A(const uint32_t seed, const int k, const int q,
      parmat_gpu<REAL> &x, parmat_gpu<REAL> &QY)
    {
      const len_global_t m = x.nrows();
      const len_t n = x.ncols();
      if (m < (len_global_t)n)
        throw std::runtime_error("must have m>n");
      if (k > n)
        throw std::runtime_error("must have k<n");
      
      auto c = x.data_obj().get_card();
      
      parmat_gpu<REAL> Y(x.get_comm(), c, m, 2*k, x.nrows_before());
      parmat_gpu<REAL> Y_tmp(x.get_comm(), c, m, 2*k, x.nrows_before());
      gpumat<REAL> Z(c, n, 2*k);
      gpumat<REAL> QZ(c, n, 2*k);
      
      gpumat<REAL> R(c, 2*k, 2*k);
      gpumat<REAL> R_local(c);
      gpuvec<REAL> qraux(c);
      gpuvec<REAL> work(c);
      
      gpumat<REAL> omega(c, n, 2*k);
      omega.fill_runif(seed);
      
      // Stage A
      matmult(x, omega, Y);
      internals::qr_Q(Y, Y_tmp, R, R_local, qraux, QY);
      
      for (int i=0; i<q; i++)
      {
        matmult(x, QY, Z);
        linalg::qr_internals(false, Z, qraux, work);
        linalg::qr_Q(Z, qraux, QZ, work);
        
        matmult(x, QZ, Y);
        internals::qr_Q(Y, Y_tmp, R, R_local, qraux, QY);
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
    
    @impl Uses a series of QR's using the TSQR implementation, and matrix
    multiplications.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rsvd(const uint32_t seed, const int k, const int q,
    parmat_gpu<REAL> &x, gpuvec<REAL> &s)
  {
    parmat_gpu<REAL> QY(x.get_comm(), s.get_card(), x.nrows(), 2*k, x.nrows_before());
    gpumat<REAL> B(s.get_card(), 2*k, x.ncols());
    
    // stage A
    internals::rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(QY, x, B);
    
    linalg::qrsvd(B, s);
    s.resize(k);
  }
  
  /// \overload
  template <typename REAL>
  void rsvd(const uint32_t seed, const int k, const int q,
    parmat_gpu<REAL> &x, gpuvec<REAL> &s, parmat_gpu<REAL> &u,
    gpumat<REAL> &vt)
  {
    parmat_gpu<REAL> QY(x.get_comm(), s.get_card(), x.nrows(), 2*k, x.nrows_before());
    gpumat<REAL> B(s.get_card(), 2*k, x.ncols());
    
    // stage A
    internals::rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(QY, x, B);
    
    cpumat<REAL> uB(s.get_card());
    linalg::qrsvd(B, s, uB, vt);
    
    s.resize(k);
    
    matmult(QY, uB, u);
    // u.resize(u.nrows(), k);
  }
}
}


#endif
