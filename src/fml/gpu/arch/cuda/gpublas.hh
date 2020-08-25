// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_GPUBLAS_H
#define FML_GPU_ARCH_CUDA_GPUBLAS_H
#pragma once


#include <cublas.h>


namespace fml
{
namespace gpublas
{
  namespace err
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
    
    inline void check_ret(cublasStatus_t check, std::string op)
    {
      if (check != CUBLAS_STATUS_SUCCESS)
      {
        std::string msg = "cuBLAS " + op + "() failed with error: " + get_cublas_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  inline cublasStatus_t set_math_mode(cublasHandle_t handle, cublasMath_t mode)
  {
    return cublasSetMathMode(handle, mode);
  }
  
  inline cublasStatus_t get_math_mode(cublasHandle_t handle, cublasMath_t *mode)
  {
    return cublasGetMathMode(handle, mode);
  }
  
  inline std::string get_math_mode_string(cublasHandle_t handle)
  {
    cublasMath_t mode;
    cublasStatus_t check = get_math_mode(handle, &mode);
    err::check_ret(check, "cublasGetMathMode");
    
    std::string ret;
    if (mode == CUBLAS_DEFAULT_MATH)
      ret = "default";
  #if __CUDACC_VER_MAJOR__ >= 11
    else if (mode == CUBLAS_PEDANTIC_MATH)
      ret = "pedantic";
    else if (mode == CUBLAS_TF32_TENSOR_OP_MATH)
      ret = "TF32 tensor op";
    else if (mode == CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
      ret = "disallow reduced precision";
  #else
    else if (mode == CUBLAS_TENSOR_OP_MATH)
      ret = "tensor op";
  #endif
    else
      throw std::runtime_error("unable to determine cuBLAS math mode");
    
    return ret;
  }
  
  
  
  inline cublasStatus_t Iamax(cublasHandle_t handle, int n, const float *x,
    int incx, int *result)
  {
    return cublasIsamax(handle, n, x, incx, result);
  }
  
  inline cublasStatus_t Iamax(cublasHandle_t handle, int n, const double *x,
    int incx, int *result)
  {
    return cublasIdamax(handle, n, x, incx, result);
  }
  
  
  
  inline cublasStatus_t Iamin(cublasHandle_t handle, int n, const float *x,
    int incx, int *result)
  {
    return cublasIsamin(handle, n, x, incx, result);
  }
  
  inline cublasStatus_t Iamin(cublasHandle_t handle, int n, const double *x,
    int incx, int *result)
  {
    return cublasIdamin(handle, n, x, incx, result);
  }
  
  
  
  inline cublasStatus_t dot(cublasHandle_t handle, int n, const float *x,
    int incx, const float *y, int incy, float *result)
  {
    return cublasSdot(handle, n, x, incx, y, incy, result);
  }
  
  inline cublasStatus_t dot(cublasHandle_t handle, int n, const double *x,
    int incx, const double *y, int incy, double *result)
  {
    return cublasDdot(handle, n, x, incx, y, incy, result);
  }
  
  
  
  inline cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const __half alpha,
    const __half *A, int lda, const __half *B, int ldb, const __half beta,
    __half *C, int ldc)
  {
    return cublasHgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
      &beta, C, ldc);
  }
  
  inline cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const float alpha,
    const float *A, int lda, const float *B, int ldb, const float beta,
    float *C, int ldc)
  {
    return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
      &beta, C, ldc);
  }
  
  inline cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, const double alpha,
    const double *A, int lda, const double *B, int ldb, const double beta,
    double *C, int ldc)
  {
    return cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
      &beta, C, ldc);
  }
  
  
  
  inline cublasStatus_t syrk(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k, const float alpha, const float *A,
    int lda, const float beta, float *C, int ldc)
  {
    return cublasSsyrk(handle, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
  }
  
  inline cublasStatus_t syrk(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k, const double alpha, const double *A,
    int lda, const double beta, double *C, int ldc)
  {
    return cublasDsyrk(handle, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
  }
  
  
  
  inline cublasStatus_t geam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, const float alpha, const float *A,
    int lda, const float beta, const float *B, int ldb, float *C, int ldc)
  {
    return cublasSgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B,
      ldb, C, ldc);
  }
  
  inline cublasStatus_t geam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, const double alpha, const double *A,
    int lda, const double beta, const double *B, int ldb, double *C, int ldc)
  {
    return cublasDgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B,
      ldb, C, ldc);
  }
  
  
  
  inline cublasStatus_t trsm(cublasHandle_t handle, cublasSideMode_t side,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const float alpha, const float *A, int lda, float *B,
    int ldb)
  {
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda,
      B, ldb);
  }
  
  inline cublasStatus_t trsm(cublasHandle_t handle, cublasSideMode_t side,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const double alpha, const double *A, int lda, double *B,
    int ldb)
  {
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda,
      B, ldb);
  }
  
  
  
  inline cublasStatus_t trmm(cublasHandle_t handle, cublasSideMode_t side,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const float alpha, const float *A, int lda, const float *B,
    int ldb, float *C, int ldc)
  {
    return cublasStrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda,
      B, ldb, C, ldc);
  }
  
  inline cublasStatus_t trmm(cublasHandle_t handle, cublasSideMode_t side,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const double alpha, const double *A, int lda, const double *B,
    int ldb, double *C, int ldc)
  {
    return cublasDtrmm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda,
      B, ldb, C, ldc);
  }
}
}


#endif
