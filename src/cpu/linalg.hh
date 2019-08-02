#ifndef FML_CPUMAT_LINALG_H
#define FML_CPUMAT_LINALG_H


#include "../linalgutils.hh"
#include "cpumat.hh"
#include "lapack.hh"


namespace linalg
{
  template <typename REAL>
  cpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, cpumat<REAL> &x, cpumat<REAL> &y)
  {
    int m, n, k;
    len_t mx = x.nrows();
    len_t my = y.nrows();
    
    linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    cpumat<REAL> ret(m, n);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void matmult_noalloc(const bool transx, const bool transy, const REAL alpha, cpumat<REAL> &x, cpumat<REAL> &y, cpumat<REAL> ret)
  {
    // if (x.ncols() != ret.nrows() || ret.ncols() != y.nrows())
    //   TODO
    
    int m, n, k;
    len_t mx = x.nrows();
    len_t my = y.nrows();
    
    linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
  }
}


#endif
