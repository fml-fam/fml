// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_CPU_LINALG_SVD_H
#define FML_PAR_CPU_LINALG_SVD_H
#pragma once


#include "../parmat.hh"

#include "blas.hh"
#include "qr.hh"

#include "../../../_internals/omp.hh"

#include "../../../cpu/linalg/linalg_blas.hh"
#include "../../../cpu/linalg/linalg_invert.hh"
#include "../../../cpu/linalg/linalg_qr.hh"
#include "../../../cpu/linalg/linalg_svd.hh"

#include "../../../cpu/copy.hh"


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
  void cpsvd(const parmat_cpu<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t n = x.ncols();
    
    auto cp = crossprod((REAL)1.0, x.data_obj());
    x.get_comm().allreduce(n*n, cp.data_ptr());
    eigen_sym(cp, s);
  }
  
  /// \overload
  template <typename REAL>
  void cpsvd(const parmat_cpu<REAL> &x, cpuvec<REAL> &s,
    cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    const len_t n = x.ncols();
    
    auto cp = crossprod((REAL)1.0, x.data_obj());
    x.get_comm().allreduce(n*n, cp.data_ptr());
    eigen_sym(cp, s, vt);
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
    
    vt.rev_cols();
    fml::copy::cpu2cpu(vt, cp);
    REAL *vt_d = vt.data_ptr();
    #pragma omp parallel for if(n*n > omp::OMP_MIN_SIZE)
    for (len_t j=0; j<n; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<n; i++)
        vt_d[i + n*j] /= s_d[j];
    }
    
    fml::linalg::matmult(false, false, (REAL)1.0, x.data_obj(), vt, u);
    fml::linalg::xpose(cp, vt);
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
  void tssvd(parmat_cpu<REAL> &x, cpuvec<REAL> &s)
  {
    cpumat<REAL> R;
    qr_R(mpi::REDUCE_TO_ALL, x, R);
    svd(R, s);
  }
  
  /// \overload
  template <typename REAL>
  void tssvd(parmat_cpu<REAL> &x, cpuvec<REAL> &s, parmat_cpu<REAL> &u,
    cpumat<REAL> &vt)
  {
    const len_t n = x.ncols();
    if (x.nrows() < (len_global_t)n)
      throw std::runtime_error("impossible dimensions");
    
    cpumat<REAL> x_cpy = x.data_obj().dupe();
    
    cpumat<REAL> R_local;
    cpuvec<REAL> qraux;
    qr(false, x.data_obj(), qraux);
    qr_R(x.data_obj(), R_local);
    
    cpumat<REAL> R(n, n);
    tsqr::qr_allreduce(mpi::REDUCE_TO_ALL, n, n, R_local.data_ptr(),
      R.data_ptr(), x.get_comm().get_comm());
    
    copy::cpu2cpu(R, R_local);
    cpumat<REAL> u_R;
    linalg::svd(R_local, s, u_R, vt);
    
    // u = Q*u_R = x*R^{-1}*u_R
    u.resize(x.nrows(), x.ncols());
    linalg::trinv(true, false, R);
    linalg::matmult(false, false, (REAL)1, x_cpy, R, x.data_obj());
    linalg::matmult(false, false, (REAL)1, x.data_obj(), u_R, u.data_obj());
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    void rsvd_A(const uint32_t seed, const int k, const int q,
      parmat_cpu<REAL> &x, parmat_cpu<REAL> &QY)
    {
      const len_global_t m = x.nrows();
      const len_t n = x.ncols();
      if (m < (len_global_t)n)
        throw std::runtime_error("must have m>n");
      if (k > n)
        throw std::runtime_error("must have k<n");
      
      parmat_cpu<REAL> Y(x.get_comm(), m, 2*k, x.nrows_before());
      parmat_cpu<REAL> Y_tmp(x.get_comm(), m, 2*k, x.nrows_before());
      cpumat<REAL> Z(n, 2*k);
      cpumat<REAL> QZ(n, 2*k);
      
      cpumat<REAL> R(2*k, 2*k);
      cpumat<REAL> R_local;
      cpuvec<REAL> qraux;
      cpuvec<REAL> work;
      
      cpumat<REAL> omega(n, 2*k);
      omega.fill_runif(seed);
      
      // Stage A
      matmult(x, omega, Y);
      qr_Q(Y, Y_tmp, R, R_local, qraux, QY);
      
      for (int i=0; i<q; i++)
      {
        matmult(x, QY, Z);
        linalg::qr_internals(false, Z, qraux, work);
        linalg::qr_Q(Z, qraux, QZ, work);
        
        matmult(x, QZ, Y);
        qr_Q(Y, Y_tmp, R, R_local, qraux, QY);
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
    parmat_cpu<REAL> &x, cpuvec<REAL> &s)
  {
    parmat_cpu<REAL> QY(x.get_comm(), x.nrows(), 2*k, x.nrows_before());
    cpumat<REAL> B(2*k, x.ncols());
    
    // stage A
    internals::rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(QY, x, B);
    
    linalg::svd(B, s);
    s.resize(k);
  }
  
  /// \overload
  template <typename REAL>
  void rsvd(const uint32_t seed, const int k, const int q,
    parmat_cpu<REAL> &x, cpuvec<REAL> &s, parmat_cpu<REAL> &u,
    cpumat<REAL> &vt)
  {
    parmat_cpu<REAL> QY(x.get_comm(), x.nrows(), 2*k, x.nrows_before());
    cpumat<REAL> B(2*k, x.ncols());
    
    // stage A
    internals::rsvd_A(seed, k, q, x, QY);
    
    // Stage B
    matmult(QY, x, B);
    
    cpumat<REAL> uB;
    linalg::svd(B, s, uB, vt);
    
    s.resize(k);
    
    matmult(QY, uB, u);
    // u.resize(u.nrows(), k);
  }
}
}


#endif
