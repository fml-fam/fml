#ifndef FML_GPU_LINALG_H
#define FML_GPU_LINALG_H


#include "../linalgutils.hh"
#include "culapack.hh"
#include "gpumat.hh"


namespace linalg
{
  template <typename REAL>
  gpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, gpumat<REAL> &x, gpumat<REAL> &y)
  {
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    auto c = x.get_card();
    gpumat<REAL> ret(c, m, n);
    
    cublasOperation_t cbtransx = transx ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cbtransy = transy ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasStatus_t check = culapack::gemm(c->cb_handle(), cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), x.nrows(), y.data_ptr(), y.nrows(), (REAL)0, ret.data_ptr(), m);
    // TODO check check
    
    return ret;
  }
}


#endif
