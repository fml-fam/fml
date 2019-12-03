// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_CUDA_CUSOLVER_H
#define FML_GPU_INTERNALS_CUDA_CUSOLVER_H
#pragma once


#include <cublas.h>
#include <cusolverDn.h>


namespace culapack
{
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
  
  
  
  inline cusolverStatus_t getrf_buflen(cusolverDnHandle_t handle, int m, int n,
     float *A, int lda, int *lwork)
  {
    return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t getrf_buflen(cusolverDnHandle_t handle, int m, int n,
     double *A, int lda, int *lwork)
  {
    return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t getrf(cusolverDnHandle_t handle, int m, int n,
     float *A, int lda, float *work, int *ipiv, int *info)
  {
    return cusolverDnSgetrf(handle, m, n, A, lda, work, ipiv, info);
  }
  
  inline cusolverStatus_t getrf(cusolverDnHandle_t handle, int m, int n,
     double *A, int lda, double *work, int *ipiv, int *info)
  {
    return cusolverDnDgetrf(handle, m, n, A, lda, work, ipiv, info);
  }
  
  
  
  inline cusolverStatus_t gesvd_buflen(cusolverDnHandle_t handle, int m, int n,
     float *A, int *lwork)
  {
    (void)A;
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
  }
  
  inline cusolverStatus_t gesvd_buflen(cusolverDnHandle_t handle, int m, int n,
     double *A, int *lwork)
  {
    (void)A;
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
  }
  
  inline cusolverStatus_t gesvd(cusolverDnHandle_t handle, signed char jobu,
    signed char jobvt, const int m, const int n, float *A, const int lda,
    float *S, float *U, const int ldu, float *VT, const int ldvt, float *work,
    const int lwork, float *rwork, int *info)
  {
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
      ldvt, work, lwork, rwork, info);
  }
  
  inline cusolverStatus_t gesvd(cusolverDnHandle_t handle, signed char jobu,
    signed char jobvt, const int m, const int n, double *A, const int lda,
    double *S, double *U, const int ldu, double *VT, const int ldvt, double *work,
    const int lwork, double *rwork, int *info)
  {
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
      ldvt, work, lwork, rwork, info);
  }
  
  
  
  inline cusolverStatus_t syevd_buflen(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A,
    int lda, const float *W, int *lwork)
  {
    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W,
      lwork);
  }
  
  inline cusolverStatus_t syevd_buflen(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A,
    int lda, const double *W, int *lwork)
  {
    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W,
      lwork);
  }
  
  inline cusolverStatus_t syevd(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda,
    float *W, float *work, int lwork, int *devInfo)
  {
    return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
      devInfo);
  }
  
  inline cusolverStatus_t syevd(cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda,
    double *W, double *work, int lwork, int *devInfo)
  {
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
      devInfo);
  }
  
  
  
  inline cublasStatus_t getri_batched(cublasHandle_t handle, const int n,
    const float **Aarray, const int lda, const int *devIpiv, float **Carray,
    const int ldb, int *info, const int batchSize)
  {
    return cublasSgetriBatched(handle, n, Aarray, lda, devIpiv, Carray, ldb,
      info, batchSize);
  }
  
  inline cublasStatus_t getri_batched(cublasHandle_t handle, const int n,
    const double **Aarray, const int lda, const int *devIpiv, double **Carray,
    const int ldb, int *info, const int batchSize)
  {
    return cublasDgetriBatched(handle, n, Aarray, lda, devIpiv, Carray, ldb,
      info, batchSize);
  }
  
  
  
  inline cusolverStatus_t getrs(cusolverDnHandle_t handle,
    cublasOperation_t trans, const int n, const int nrhs, const float *A,
    const int lda, const int *devIpiv, float *B, const int ldb, int *info)
  {
    return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
      info);
  }
  
  inline cusolverStatus_t getrs(cusolverDnHandle_t handle,
    cublasOperation_t trans, const int n, const int nrhs, const double *A,
    const int lda, const int *devIpiv, double *B, const int ldb, int *info)
  {
    return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
      info);
  }
}


#endif
