#ifndef FML_MPI_LINALG_H
#define FML_MPI_LINALG_H


#include <stdexcept>

#include "../linalgutils.hh"
#include "../cpu/cpuvec.hh"

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
  
  
  
  // upper triangle
  template <typename REAL>
  mpimat<REAL> crossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    len_t n = x.ncols();
    grid g = x.get_grid();
    
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    ret.fill_zero();
    
    scalapack::syrk('L', 'T', n, x.nrows(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
    
    return ret;
  }
  
  template <typename REAL>
  void crossprod_noalloc(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    len_t n = x.ncols();
    if (n != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    scalapack::syrk('L', 'T', n, x.nrows(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  
  
  template <typename REAL>
  mpimat<REAL> tcrossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    len_t n = x.nrows();
    grid g = x.get_grid();
    
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    ret.fill_zero();
    
    scalapack::syrk('L', 'N', n, x.ncols(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
    
    return ret;
  }
  
  template <typename REAL>
  void tcrossprod_noalloc(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    len_t n = x.nrows();
    if (n != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    scalapack::syrk('L', 'N', n, x.ncols(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  
  
  template <typename REAL>
  int lu(mpimat<REAL> &x, cpuvec<int> &p)
  {
    int info = 0;
    len_t m = x.nrows();
    len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    scalapack::getrf(m, x.ncols(), x.data_ptr(), m, p.data_ptr(), &info);
    
    return info;
  }
  
  template <typename REAL>
  int lu(mpimat<REAL> &x)
  {
    cpuvec<int> p;
    return lu(x, p);
  }
}


#endif
