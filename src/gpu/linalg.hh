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
        return "unknown cuBLAS error occurred";
    }
    
    inline void check_cublas_ret(cublasStatus_t check, std::string op)
    {
      if (check != CUBLAS_STATUS_SUCCESS)
      {
        std::string msg = "cuBLAS " + op + "() failed with error: " + get_cublas_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
    
    
    
    inline std::string get_cusolver_error_msg(cusolverStatus_t check)
    {
      if (check == CUSOLVER_STATUS_SUCCESS)
        return "";
      else if (check == CUSOLVER_STATUS_NOT_INITIALIZED)
        return "cuSOLVER not initialized";
      else if (check == CUSOLVER_STATUS_ALLOC_FAILED)
        return "internal cuSOLVER memory allocation failed";
      else if (check == CUSOLVER_STATUS_INVALID_VALUE)
        return "unsupported parameter";
      else if (check == CUSOLVER_STATUS_ARCH_MISMATCH)
        return "function requires feature missing from device architecture";
      else if (check == CUSOLVER_STATUS_EXECUTION_FAILED)
        return "GPU program failed to execute";
      else if (check == CUSOLVER_STATUS_INTERNAL_ERROR)
        return "internal cuSOLVER operation failed";
      else if (check == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
        return "matrix type not supported";
      else
        return "unknown cuSOLVER error occurred";
    }
    
    inline void check_cusolver_ret(cusolverStatus_t check, std::string op)
    {
      if (check != CUSOLVER_STATUS_SUCCESS)
      {
        std::string msg = "cuSOLVER " + op + "() failed with error: " + get_cusolver_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  /**
   * @brief Returns alpha*op(x)*op(y) where op(A) is A or A^T
   * 
   * @param[in] transx Should x^T be used?
   * @param[in] transy Should y^T be used?
   * @param[in] alpha Scalar.
   * @param[in] x Left multiplicand.
   * @param[in] y Right multiplicand.
   * 
   * @details If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. Likewise for ret.
   * 
   * @impl Uses the cuBLAS function cublasXgemm().
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  gpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const gpumat<REAL> &x, const gpumat<REAL> &y)
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
  
  /**
   * @brief Computes ret = alpha*op(x)*op(y) where op(A) is A or A^T
   * 
   * @param[in] transx Should x^T be used?
   * @param[in] transy Should y^T be used?
   * @param[in] alpha Scalar.
   * @param[in] x Left multiplicand.
   * @param[in] y Right multiplicand.
   * @param[out] ret The product.
   * 
   * @details If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. Likewise for ret.
   * 
   * @impl Uses the cuBLAS function cublasXgemm().
   * 
   * @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const gpumat<REAL> &x, const gpumat<REAL> &y, gpumat<REAL> &ret)
  {
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
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
  
  
  
  template <typename REAL>
  int lu(gpumat<REAL> &x, gpuvec<int> &p)
  {
    int info = 0;
    const int m = x.nrows();
    auto c = x.get_card();
    
    const len_t lipiv = std::min(m, x.ncols());
    if (!p.get_card()->valid_card())
      p.set(c);
    
    p.resize(lipiv);
    
    int lwork;
    cusolverStatus_t check = culapack::getrf_buflen(c->cs_handle(), m, m, x.data_ptr(), m, &lwork);
    check_cusolver_ret(check, "getrf_bufferSize");
    
    REAL *work = (REAL*) c->mem_alloc(lwork*sizeof(*work));
    int *info_device = (int*) c->mem_alloc(sizeof(*info_device));
    
    c->mem_cpu2gpu(info_device, &info, sizeof(info));
    check = culapack::getrf(c->cs_handle(), m, m, x.data_ptr(), m, work, p.data_ptr(), info_device);
    c->mem_gpu2cpu(&info, info_device, sizeof(info));
    
    c->mem_free(work);
    c->mem_free(info_device);
    
    check_cusolver_ret(check, "getrf");
    
    return info;
  }
  
  template <typename REAL>
  int lu(gpumat<REAL> &x)
  {
    gpuvec<int> p(x.get_card());
    return lu(x, p);
  }
}


#endif
