#ifndef FML_MPIMAT_LINALG_H
#define FML_MPIMAT_LINALG_H


#include <stdexcept>

#include "../linalgutils.hh"
#include "mpimat.hh"
#include "scalapack.hh"


namespace linalg
{
  template <typename REAL>
  mpimat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
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
  void matmult_noalloc(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    if (x.ncols() != ret.nrows() || ret.ncols() != y.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
  }
}


#endif
