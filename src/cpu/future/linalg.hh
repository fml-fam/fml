// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_FUTURE_LINALG_H
#define FML_CPU_FUTURE_LINALG_H
#pragma once


#include "../linalg.hh"


namespace linalg
{
  template <typename REAL>
  void rsvd(const int k, const int q, cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    cpumat<REAL> omega(n, 2*k);
    omega.fill_runif();
    
    cpumat<REAL> Y(m, 2*k);
    cpumat<REAL> QY(m, 2*k);
    cpumat<REAL> Z(n, 2*k);
    cpumat<REAL> QZ(n, 2*k);
    
    cpuvec<REAL> qraux;
    cpuvec<REAL> work;
    
    cpumat<REAL> B(2*k, n);
    
    // Stage A
    matmult(false, false, (REAL)1.0, x, omega, Y);
    qr_internals(false, Y, qraux, work);
    qr_Q(Y, qraux, QY, work);
    
    for (int i=1; i<q; i++)
    {
      matmult(true, false, (REAL)1.0, x, QY, Z);
      qr_internals(false, Z, qraux, work);
      qr_Q(Z, qraux, QZ, work);
      
      matmult(false, false, (REAL)1.0, x, QZ, Y);
      qr_internals(false, Y, qraux, work);
      qr_Q(Y, qraux, QY, work);
    }
    
    // Stage B
    matmult(true, false, (REAL)1.0, QY, x, B);
    
    cpumat<REAL> uB;
    svd(B, s, uB, vt);
    
    s.resize(k);
    
    matmult(false, false, (REAL)1.0, QY, uB, u);
    u.resize(u.nrows(), k);
  }
}


#endif
