#ifndef FML_GPUMAT_CUSOLVER_H
#define FML_GPUMAT_CUSOLVER_H


#include <cusolverDn.h>


namespace lapack
{
  cublasStatus_t cu_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
  {
    cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  
  cublasStatus_t cu_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
  {
    cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  
  
  
  cusolverStatus_t cu_getrf_buflen(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *lwork)
  {
    return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  cusolverStatus_t cu_getrf_buflen(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *lwork)
  {
    return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  cusolverStatus_t cu_getrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *work, int *ipiv, int *info)
  {
    return cusolverDnSgetrf(handle, m, n, A, lda, work, ipiv, info);
  }
  
  cusolverStatus_t cu_getrf(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *work, int *ipiv, int *info)
  {
    return cusolverDnDgetrf(handle, m, n, A, lda, work, ipiv, info);
  }
}


#endif
