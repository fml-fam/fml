// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_LINALG_QR_H
#define FML_PAR_GPU_LINALG_QR_H
#pragma once


#include "../parmat.hh"

#include "../../../gpu/linalg/linalg_invert.hh"
#include "../../../gpu/linalg/linalg_qr.hh"

#include "../../../gpu/copy.hh"

#include "blas.hh"
#include "qr_allreduce.hh"


namespace fml
{
namespace linalg
{
  namespace internals
  {
    template <typename REAL>
    void qr_R(const int root, parmat_gpu<REAL> &x, gpumat<REAL> &R,
      gpumat<REAL> &R_local, gpuvec<REAL> &qraux)
    {
      const len_t n = x.ncols();
      
      linalg::qr(false, x.data_obj(), qraux);
      linalg::qr_R(x.data_obj(), R_local);
      
      R.resize(n, n);
      tsqr::qr_allreduce(root, n, n, R_local.data_ptr(), R.data_ptr(),
        x.get_comm().get_comm(), R.get_card());
    }
  }
  
  
  
  template <typename REAL>
  void qr_R(const int root, parmat_gpu<REAL> &x, gpumat<REAL> &R)
  {
    if (x.nrows() < (len_global_t)x.ncols())
      throw std::runtime_error("impossible dimensions");
    
    gpumat<REAL> R_local(R.get_card());
    gpuvec<REAL> qraux(R.get_card());
    
    internals::qr_R(root, x, R, R_local, qraux);
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    void qr_Q(const parmat_gpu<REAL> &x, parmat_gpu<REAL> &x_cpy,
      gpumat<REAL> &R, gpumat<REAL> &R_local, gpuvec<REAL> &qraux,
      parmat_gpu<REAL> &Q)
    {
      copy::gpu2gpu(x.data_obj(), x_cpy.data_obj());
      internals::qr_R(mpi::REDUCE_TO_ALL, x_cpy, R, R_local, qraux);
      trinv(true, false, R);
      matmult(x, R, Q);
    }
  }
  
  template <typename REAL>
  void qr_Q(parmat_gpu<REAL> &x, gpuvec<REAL> &qraux, parmat_gpu<REAL> &Q)
  {
    gpumat<REAL> R, R_local;
    
    qr_R(mpi::REDUCE_TO_ALL, x, R, R_local, qraux);
    trinv(true, false, R);
    matmult(x, R, Q);
  }
}
}


#endif
