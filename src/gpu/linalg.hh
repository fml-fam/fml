#ifndef FML_GPU_LINALG_H
#define FML_GPU_LINALG_H


#include <stdexcept>

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
    if (check != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("asdf");
    
    return ret;
  }
  
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, gpumat<REAL> &x, gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    cublasOperation_t cbtransx = transx ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cbtransy = transy ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    cublasStatus_t check = culapack::gemm(x.get_card()->cb_handle(), cbtransx, cbtransy, m, n, k, alpha, x.data_ptr(), x.nrows(), y.data_ptr(), y.nrows(), (REAL)0, ret.data_ptr(), m);
    if (check != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("asdf");
  }
  
  
  
  // lower triangle of t(x) %*% x
  template <typename REAL>
  void crossprod(const REAL alpha, const gpumat<REAL> &x, gpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    auto cbh = x.get_card()->cb_handle();
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    
    culapack::syrk(cbh, uplo, trans, n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
  }
  
  template <typename REAL>
  gpumat<REAL> crossprod(const REAL alpha, const gpumat<REAL> &x)
  {
    const len_t n = x.ncols();
    auto c = x.get_card();
    
    gpumat<REAL> ret(c, n, n);
    ret.fill_zero();
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void tcrossprod(const REAL alpha, const gpumat<REAL> &x, gpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != ret.nrows() || m != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    auto cbh = x.get_card()->cb_handle();
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    
    culapack::syrk(cbh, uplo, trans, m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
  }
  
  template <typename REAL>
  gpumat<REAL> tcrossprod(const REAL alpha, const gpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    auto c = x.get_card();
    
    gpumat<REAL> ret(c, m, m);
    ret.fill_zero();
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
}


#endif
