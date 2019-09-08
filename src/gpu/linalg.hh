#ifndef FML_GPU_LINALG_H
#define FML_GPU_LINALG_H


#include <stdexcept>
#include <string>

#include "../linalgutils.hh"
#include "culapack.hh"
#include "gpumat.hh"


namespace linalg
{
  namespace
  {
    inline std::string get_cublas_error_msg(cublasStatus_t check)
    {
      if (check == CUBLAS_STATUS_SUCCESS)
        return "";
      else if (check == CUBLAS_STATUS_NOT_INITIALIZED)
        return "cuBLAS not initialized";
      else if (check == CUBLAS_STATUS_ALLOC_FAILED)
        return "internal cuBLAS memory allocation failed";
      else if (check == CUBLAS_STATUS_INVALID_VALUE)
        return "unsupported parameter";
      else if (check == CUBLAS_STATUS_ARCH_MISMATCH)
        return "function requires feature missing from device architecture";
      else if (check == CUBLAS_STATUS_MAPPING_ERROR)
        return "access to GPU memory space failed";
      else if (check == CUBLAS_STATUS_EXECUTION_FAILED)
        return "GPU program failed to execute";
      else if (check == CUBLAS_STATUS_INTERNAL_ERROR)
        return "internal cuBLAS operation failed";
      else if (check == CUBLAS_STATUS_NOT_SUPPORTED)
        return "requested functionality is not supported";
      else if (check == CUBLAS_STATUS_LICENSE_ERROR)
        return "error with cuBLAS license check";
      else
        return "error unknown to fml developers";
    }
    
    inline void check_cublas_ret(cublasStatus_t check, std::string op)
    {
      if (check != CUBLAS_STATUS_SUCCESS)
      {
        std::string msg = "cuBLAS " + op + "() failed with error: " + get_cublas_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
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
    check_cublas_ret(check, "gemm");
    
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
    check_cublas_ret(check, "gemm");
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
    
    cublasStatus_t check = culapack::syrk(cbh, uplo, trans, n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
    check_cublas_ret(check, "syrk");
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
    
    cublasStatus_t check = culapack::syrk(cbh, uplo, trans, m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
    check_cublas_ret(check, "syrk");
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
  
  
  
  template <typename REAL>
  void xpose(const gpumat<REAL> &x, gpumat<REAL> &tx)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    auto cbh = x.get_card()->cb_handle();
    
    cublasStatus_t check = culapack::geam(cbh, CUBLAS_OP_T, CUBLAS_OP_N, n, m, (REAL)1.0, x.data_ptr(), m, (REAL) 0.0, tx.data_ptr(), n, tx.data_ptr(), n);
    check_cublas_ret(check, "geam");
  }
  
  template <typename REAL>
  gpumat<REAL> xpose(const gpumat<REAL> &x)
  {
    gpumat<REAL> tx(x.get_card(), x.ncols(), x.nrows());
    xpose(x, tx);
    return tx;
  }
}


#endif
