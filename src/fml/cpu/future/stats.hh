// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_FUTURE_STATS_H
#define FML_CPU_FUTURE_STATS_H
#pragma once


#include "../linalg.hh"


namespace stats
{
  // linalg
  template <typename REAL>
  REAL norm_1(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0.0;
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m; i++)
        norm += fabs(x_d[i + m*j]);
    }
    
    return norm;
  }
  
  template <typename REAL>
  REAL norm_F(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0.0;
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m; i++)
        norm += x_d[i + m*j] * x_d[i + m*j];
    }
    
    return norm;
  }
  
  
  
  // dimops
  template <typename REAL>
  void rowprod(cpumat<REAL> &x, const cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (s.size() != n)
      throw std::runtime_error("incompatible dimensions");
    
    const REAL *x_d = x.data_ptr();
    const REAL *s_d = s.data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<m; i++)
        x_d[i + m*j] *= s_d[j];
    }
  }
  
  template <typename REAL>
  void colprod(cpumat<REAL> &x, const cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (s.size() != m)
      throw std::runtime_error("incompatible dimensions");
    
    const REAL *x_d = x.data_ptr();
    const REAL *s_d = s.data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<m; i++)
        x_d[i + m*j] *= s_d[i];
    }
  }
  
  
  
  // ???
  template <typename T>
  T sgn(T x)
  {
    return (T>0?1:(T<0?-1:0));
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void ambpc(cpumat<REAL> &a, const cpumat<REAL> &b, const cpumat<REAL> &c)
    {
      const len_t m = a.nrows();
      const len_t n = a.ncols();
      
      REAL *a_d = a.data_ptr();
      REAL *b_d = b.data_ptr();
      REAL *c_d = c.data_ptr();
      
      for (len_t j=0; j<n; j++)
      {
        for (len_t i=0; i<m; i++)
          a_d[i + m*j] = a_d[i + m*j] - b[i + m*j] + c[i + m*j];
      }
    }
    
    template <typename REAL>
    void shrink_op(cpuvec<REAL> &x, const REAL tau)
    {
      const len_t n = x.size();
      const REAL *x_d = x.data_ptr();
      
      for (len_t i=0; i<n; i++)
        x_d[i] = sgn(x[i]) * std::max(fabs(x[i]) - tau, 0);
    }
    
    template <typename REAL>
    void shrink_op(cpumat<REAL> &x, const REAL tau)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const REAL *x_d = x.data_ptr();
      
      for (len_t j=0; j<n; j++)
      {
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] = sgn(x[i + m*j]) * std::max(fabs(x[i + m*j]) - tau, 0);
      }
    }
    
    template <typename REAL>
    void sv_thresh(cpumat<REAL> &x, const REAL tau, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
    {
      linalg::svd(x, s, u, vt);
      shrink_op(s, tau);
      
      colprod(vt, s);
      matmult(false, false, (REAL)1.0, u, vt, x);
    }
  }
  
  template <typename REAL>
  void robpca(cpumat<REAL> &M, const REAL delta=1e-7, const uint32_t maxiter=1000)
  {
    const len_t n1 = M.nrows();
    const len_t n2 = M.ncols();
    
    const REAL lambda = 1/sqrt((REAL)std::max(n1, n2))
    const REAL mu = 0.25 * ((REAL)n1*n2) / norm_1(M);
    
    cpumat<REAL> S(n1, n2);
    cpumat<REAL> Y(n1, n2);
    cpumat<REAL> L(n1, n2);
    
    S.fill_zero();
    Y.fill_zero();
    
    cpuvec<REAL> s;
    cpumat<REAL> u;
    cpumat<REAL> vt;
    
    bool conv = false;
    uint32_t iter = 0;
    REAL term;
    
    REAL ub = delta * norm_F(M);
    
    while (!conv && iter<maxiter)
    {
      copy::cpu2cpu(M, L);
      if (iter > 0)
        ambpc(L, S, Y);
      
      sv_thresh(L, 1/mu, s, u, vt);
      
      // S = M - L + Y
      copy::cpu2cpu(M, S);
      ambpc(S, L, Y);
      
      sv_thresh(S, lambda/mu, s, u, vt);
      
      // Y = M - L + Y
      
      
      
      term = norm_F(...);
      conv = (term <= ub);
      iter++;
    }
  }
}


#endif
