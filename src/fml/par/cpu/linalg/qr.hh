// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_CPU_LINALG_QR_H
#define FML_PAR_CPU_LINALG_QR_H
#pragma once


#include "../parmat.hh"

#include "../../../cpu/linalg/linalg_blas.hh"
#include "../../../cpu/linalg/linalg_invert.hh"
#include "../../../cpu/linalg/linalg_qr.hh"

#include "qr_allreduce.hh"


namespace fml
{
namespace linalg
{
  namespace internals
  {
    template <typename REAL>
    void qr_R(const int root, parmat_cpu<REAL> &x, cpumat<REAL> &R,
      cpumat<REAL> &R_local, cpuvec<REAL> &qraux)
    {
      const len_t n = x.ncols();
      
      linalg::qr(false, x.data_obj(), qraux);
      linalg::qr_R(x.data_obj(), R_local);
      
      R.resize(n, n);
      tsqr::qr_allreduce(root, n, n, R_local.data_ptr(), R.data_ptr(),
        x.get_comm().get_comm());
    }
  }
  
  template <typename REAL>
  void qr_R(const int root, parmat_cpu<REAL> &x, cpumat<REAL> &R)
  {
    if (x.nrows() < (len_global_t) x.ncols())
      throw std::runtime_error("impossible dimensions");
    
    cpumat<REAL> R_local;
    cpuvec<REAL> qraux;
    
    internals::qr_R(root, x, R, R_local, qraux);
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    void qr_Q(const parmat_cpu<REAL> &x, parmat_cpu<REAL> &x_cpy,
      cpumat<REAL> &R, cpumat<REAL> &R_local,
      cpuvec<REAL> &qraux, parmat_cpu<REAL> &Q)
    {
      copy::cpu2cpu(x.data_obj(), x_cpy.data_obj());
      qr_R(mpi::REDUCE_TO_ALL, x_cpy, R, R_local, qraux);
      linalg::trinv(true, false, R);
      matmult(x, R, Q);
    }
  }
  
  template <typename REAL>
  void qr_Q(parmat_cpu<REAL> &x, cpuvec<REAL> &qraux, parmat_cpu<REAL> &Q)
  {
    cpumat<REAL> R, R_local;
    
    qr_R(mpi::REDUCE_TO_ALL, x, R, R_local, qraux);
    linalg::trinv(true, false, R);
    matmult(x, R, Q);
  }
}
}


#endif
