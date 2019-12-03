// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_FUTURE_LINALG_H
#define FML_CPU_FUTURE_LINALG_H
#pragma once


#include "../linalg.hh


namespace linalg
{
  template <typename REAL>
  void det(cpumat<REAL> &x, int &sign, REAL &modulus)
  {
    const len_t m = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    cpuvec<int> p;
    int info = lu(x, p);
    
    if (info != 0)
    {
      if (info > 0)
      {
        sign = 1;
        modulus = -INFINITY;
        return;
      }
      else
        return;
    }
    
    
    // get determinant
    REAL mod = 0.0;
    int sgn = 1;
    
    const int *ipiv = p.data_ptr();
    for (int i=0; i<m; i++)
    {
      if (ipiv[i] != (i + 1))
        sgn = -sgn;
    }
    
    const REAL *a = x.data_ptr();
    
    #pragma omp parallel for reduction(+:mod) reduction(*:sgn)
    for (int i=0; i<m; i+=m+1)
    {
      const REAL d = a[i + m*i];
      if (d < 0)
      {
        mod += log(-d);
        sgn *= -1;
      }
      else
        mod += log(d);
    }
    
    modulus = mod;
    sign = sgn;
  }
  
  
  
  namespace qr
  {
    template <typename REAL>
    void qr(cpumat<REAL> &x, cpuvec<REAL> &qraux)
    {
      len_t m = x.nrows();
      len_t n = x.ncols();
      len_t minmn = std::min(m, n);
      
      qraux.resize(minmn);
      
      REAL tmp;
      geqp3(m, n, 0.f, m, 0, 0.f, &tmp, -1, 0);
      int lwork = std::max((int) tmp, 1);
      cpuvec<REAL> work(lwork);
      
      cpuvec<int> pivot(n);
      pivot.fill_zero();
      
      int info = 0;
      geqp3(m, n, x.data_ptr(), m, pivot.data_ptr(), qraux.data_ptr(), work.data_ptr(), lwork, &info);
      
      if (info != 0)
        error("sgeqp3() returned info=%d\n", info);
    }
    
    template <typename REAL>
    void q(const cpumat<REAL> &QR, const cpuvec<REAL> &qraux, cpumat<REAL> &Q)
    {
      len_t m = QR.nrows();
      len_t n = QR.ncols();
      
      len_t nrhs = std::min(m, n);
      Q.resize(m, nrhs);
      Q.fill_zero();
      
      REAL *Q_d = Q.data_ptr();
      for (len_t i=0; i<m*nrhs; i+=m+1)
        Q_d[i] = 1.f;
      
      int info = 0;
      float tmp;
      int lwork = -1;
      ormqr('L', 'N', m, nrhs, n, QR.data_ptr(), m, qraux.data_ptr(), Q_d, m, &tmp, lwork, &info);
      lwork = (int) tmp;
      cpumat<REAL> work(lwork);
      
      ormqr('L', 'N', m, nrhs, n, QR.data_ptr(), m, qraux.data_ptr(), Q_d, m, work.data_ptr(), lwork, &info);
      
      if (info != 0)
        error("sormqr() returned info=%d\n", info);
    }
  }
}


#endif
