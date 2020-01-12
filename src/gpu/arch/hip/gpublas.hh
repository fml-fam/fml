// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_HIP_GPUBLAS_H
#define FML_GPU_ARCH_HIP_GPUBLAS_H
#pragma once


#include <rocblas.h>


namespace gpublas
{
  namespace err
  {
    inline std::string get_rocblas_error_msg(cublasStatus_t check)
    {
      if (check == rocblas_status_success)
        return "";
      else if (check == rocblas_status_invalid_handle)
        return "invalid handle";
      else if (check == rocblas_status_not_implemented)
        return "function not implemented";
      else if (check == rocblas_status_invalid_pointer)
        return "invalid data";
      else if (check == rocblas_status_invalid_size)
        return "invalid size";
      else if (check == rocblas_status_memory_error)
        return "failed internal memory operation";
      else if (check == rocblas_status_internal_error)
        return "internal library failure";
      else if (check == rocblas_status_perf_degraded)
        return "performance degraded from low device memory";
      else if (check == rocblas_status_size_query_mismatch)
        return "unmatched start/stop size query";
      else if (check == rocblas_status_size_increased)
        return "queried device memory size increased";
      else if (check == rocblas_status_size_unchanged)
        return "queried device memory size unchanged";
      else if (check == rocblas_status_invalid_value)
        return "invalid paramater";
      else if (check == rocblas_status_continue)
        return "nothing preventing function to proceed";
      else
        return "unknown rocblas error occurred";
    }
    
    inline void check_ret(rocblas_status check, std::string op)
    {
      if (check != rocblas_status_success)
      {
        std::string msg = "rocblas " + op + "() failed with error: " + get_rocblas_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  inline rocblas_status gemm(rocblas_handle handle, rocblas_operation transa,
    rocblas_operation transb, int m, int n, int k, const __half alpha,
    const __half *A, int lda, const __half *B, int ldb, const __half beta,
    __half *C, int ldc)
  {
    return rocblas_hgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
      &beta, C, ldc);
  }
  
  inline rocblas_status gemm(rocblas_handle handle, rocblas_operation transa,
    rocblas_operation transb, int m, int n, int k, const float alpha,
    const float *A, int lda, const float *B, int ldb, const float beta,
    float *C, int ldc)
  {
    return rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
      &beta, C, ldc);
  }
  
  inline rocblas_status gemm(rocblas_handle handle, rocblas_operation transa,
    rocblas_operation transb, int m, int n, int k, const double alpha,
    const double *A, int lda, const double *B, int ldb, const double beta,
    double *C, int ldc)
  {
    return rocblas_dgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
      &beta, C, ldc);
  }
  
  
  
  inline rocblas_status syrk(rocblas_handle handle, cublasFillMode_t uplo,
    rocblas_operation trans, int n, int k, const float alpha, const float *A,
    int lda, const float beta, float *C, int ldc)
  {
    return rocblas_ssyrk(handle, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
  }
  
  inline rocblas_status syrk(rocblas_handle handle, cublasFillMode_t uplo,
    rocblas_operation trans, int n, int k, const double alpha, const double *A,
    int lda, const double beta, double *C, int ldc)
  {
    return rocblas_dsyrk(handle, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
  }
  
  
  
  inline rocblas_status geam(rocblas_handle handle, rocblas_operation transa,
    rocblas_operation transb, int m, int n, const float alpha, const float *A,
    int lda, const float beta, const float *B, int ldb, float *C, int ldc)
  {
    return rocblas_sgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B,
      ldb, C, ldc);
  }
  
  inline rocblas_status geam(rocblas_handle handle, rocblas_operation transa,
    rocblas_operation transb, int m, int n, const double alpha, const double *A,
    int lda, const double beta, const double *B, int ldb, double *C, int ldc)
  {
    return rocblas_dgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B,
      ldb, C, ldc);
  }
}


#endif
