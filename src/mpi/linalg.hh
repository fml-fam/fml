#ifndef FML_MPIMAT_LINALG_H
#define FML_MPIMAT_LINALG_H


#include "mpimat.hh"
#include "scalapack.hh"


namespace linalg
{
  namespace
  {
    template <typename REAL>
    void matmult_params(const bool transx, const bool transy, const mpimat<REAL> &x, const mpimat<REAL> &y, int *m, int *n, int *k)
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
  mpimat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, mpimat<REAL> &x, mpimat<REAL> &y)
  {
    int m, n, k;
    matmult_params(transx, transy, x, y, &m, &n, &k);
    grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void matmult_noalloc(const bool transx, const bool transy, const REAL alpha, mpimat<REAL> &x, mpimat<REAL> &y, mpimat<REAL> ret)
  {
    // if (x.ncols() != ret.nrows() || ret.ncols() != y.nrows())
    //   TODO
    
    int m, n, k;
    matmult_params(transx, transy, x, y, &m, &n, &k);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
  }
}


#endif
