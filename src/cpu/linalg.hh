#ifndef FML_CPUMAT_LINALG_H
#define FML_CPUMAT_LINALG_H


#include "cpumat.hh"
#include "lapack.hh"


namespace linalg
{
  namespace
  {
    template <typename REAL>
    void matmult_params(const bool transx, const bool transy, const cpumat<REAL> &x, const cpumat<REAL> &y, int *m, int *n, int *k)
    {
      // m = # rows of op(x)
      // n = # cols of op(y)
      // k = # cols of op(x)
      
      int mx = x.nrows();
      int nx = x.ncols();
      int my = y.nrows();
      int ny = y.ncols();
      
      if (transx)
      {
        *m = nx;
        *k = mx;
      }
      else
      {
        *m = mx;
        *k = nx;
      }
      
      *n = transy ? my : ny;
    }
  }
  
  
  
  template <typename REAL>
  cpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, cpumat<REAL> &x, cpumat<REAL> &y)
  {
    int m, n, k;
    matmult_params(transx, transy, x, y, &m, &n, &k);
    cpumat<REAL> ret(m, n);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    int mx = x.nrows();
    int my = y.nrows();
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void matmult_noalloc(const bool transx, const bool transy, const REAL alpha, cpumat<REAL> &x, cpumat<REAL> &y, cpumat<REAL> ret)
  {
    // if (x.ncols() != ret.nrows() || ret.ncols() != y.nrows())
    //   TODO
    
    int m, n, k;
    matmult_params(transx, transy, x, y, &m, &n, &k);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    int mx = x.nrows();
    int my = y.nrows();
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
  }
}


#endif
