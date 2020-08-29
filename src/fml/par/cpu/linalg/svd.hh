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
}
}


#endif
