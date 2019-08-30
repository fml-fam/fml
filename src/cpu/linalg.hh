#ifndef FML_CPU_LINALG_H
#define FML_CPU_LINALG_H


#include <cmath>
#include <stdexcept>

#include "../linalgutils.hh"
#include "../omputils.hh"

#include "cpumat.hh"
#include "cpuvec.hh"
#include "lapack.hh"


namespace linalg
{
  template <typename REAL>
  cpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    int m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    cpumat<REAL> ret(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
    
    return ret;
  }
  
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const cpumat<REAL> &x, const cpumat<REAL> &y, cpumat<REAL> &ret)
  {
    if (x.nrows() != ret.nrows() || ret.ncols() != y.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    int m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
  }
  
  
  
  // lower triangle of t(x) %*% x
  template <typename REAL>
  cpumat<REAL> crossprod(const REAL alpha, const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    cpumat<REAL> ret(n, n);
    ret.fill_zero();
    
    lapack::syrk('L', 'T', n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
    
    return ret;
  }
  
  template <typename REAL>
  void crossprod(const REAL alpha, const cpumat<REAL> &x, cpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    lapack::syrk('L', 'T', n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
  }
  
  
  
  template <typename REAL>
  cpumat<REAL> tcrossprod(const REAL alpha, const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    cpumat<REAL> ret(m, m);
    ret.fill_zero();
    
    lapack::syrk('L', 'N', m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
    
    return ret;
  }
  
  template <typename REAL>
  void tcrossprod(const REAL alpha, const cpumat<REAL> &x, cpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != ret.nrows() || m != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    lapack::syrk('L', 'N', m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
  }
  
  
  
  template <typename REAL>
  int lu(cpumat<REAL> &x, cpuvec<int> &p)
  {
    int info = 0;
    const len_t m = x.nrows();
    const len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    lapack::getrf(m, x.ncols(), x.data_ptr(), m, p.data_ptr(), &info);
    
    return info;
  }
  
  template <typename REAL>
  int lu(cpumat<REAL> &x)
  {
    cpuvec<int> p;
    return lu(x, p);
  }
  
  
  
  template <typename REAL>
  void det(cpumat<REAL> &x, int &sign, REAL &modulus)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m != n)
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
  
  
  
  namespace
  {
    template <typename REAL>
    int svd_internals(const int nu, const int nv, cpumat<REAL> &x, cpuvec<REAL> &s,
      cpumat<REAL> &u, cpumat<REAL> &vt)
    {
      int info = 0;
      char jobz;
      int ldvt;
      
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      s.resize(minmn);
      
      if (nu == 0 && nv == 0)
      {
        jobz = 'N';
        ldvt = 1; // value is irrelevant, but must exist!
      }
      else if (nu <= minmn && nv <= minmn)
      {
        jobz = 'S';
        ldvt = minmn;
        
        u.resize(m, minmn);
        vt.resize(minmn, n);
      }
      else
      {
        jobz = 'A';
        ldvt = n;
      }
      
      cpuvec<int> iwork(8*minmn);
      
      REAL tmp;
      lapack::gesdd(jobz, m, n, x.data_ptr(), m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), ldvt, &tmp, -1, iwork.data_ptr(), &info);
      int lwork = (int) tmp;
      cpuvec<REAL> work(lwork);
      
      lapack::gesdd(jobz, m, n, x.data_ptr(), m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), ldvt, work.data_ptr(), lwork, iwork.data_ptr(), &info);
      
      return info;
    }
  }
  
  template <typename REAL>
  void svd(cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    cpumat<REAL> ignored;
    svd_internals(0, 0, x, s, ignored, ignored);
  }
  
  template <typename REAL>
  void svd(cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    svd_internals(1, 1, x, s, u, vt);
  }
}


#endif
