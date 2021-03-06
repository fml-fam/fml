// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_GPULAPACK_H
#define FML_GPU_ARCH_CUDA_GPULAPACK_H
#pragma once


#include <cublas.h>
#include <cusolverDn.h>


namespace fml
{
namespace gpulapack
{
  namespace err
  {
    inline std::string get_cusolver_error_msg(cusolverStatus_t check)
    {
      // if (check == CUSOLVER_STATUS_SUCCESS)
      //   return "";
      if (check == CUSOLVER_STATUS_NOT_INITIALIZED)
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
    
    inline void check_ret(cusolverStatus_t check, std::string op)
    {
      if (check != CUSOLVER_STATUS_SUCCESS)
      {
        std::string msg = "cuSOLVER " + op + "() failed with error: " + get_cusolver_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
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
  
  
  
  inline cusolverStatus_t geqrf_buflen(cusolverDnHandle_t handle,
    const int m, const int n, float *A, const int lda, int *lwork)
  {
    return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t geqrf_buflen(cusolverDnHandle_t handle,
    const int m, const int n, double *A, const int lda, int *lwork)
  {
    return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t geqrf(cusolverDnHandle_t handle, const int m,
    const int n, float *A, const int lda, float *tau, float *work,
    const int lwork, int *info)
  {
    return cusolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
  }
  
  inline cusolverStatus_t geqrf(cusolverDnHandle_t handle, const int m,
    const int n, double *A, const int lda, double *tau, double *work,
    const int lwork, int *info)
  {
    return cusolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
  }
  
  
  
  inline cusolverStatus_t ormqr_buflen(cusolverDnHandle_t handle,
    cublasSideMode_t side, cublasOperation_t trans, const int m, const int n,
    const int k, const float *A, const int lda, const float *tau,
    const float *C, const int ldc, int *lwork)
  {
    return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda,
      tau, C, ldc, lwork);
  }
  
  inline cusolverStatus_t ormqr_buflen(cusolverDnHandle_t handle,
    cublasSideMode_t side, cublasOperation_t trans, const int m, const int n,
    const int k, const double *A, const int lda, const double *tau,
    const double *C, const int ldc, int *lwork)
  {
    return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda,
      tau, C, ldc, lwork);
  }
  
  inline cusolverStatus_t ormqr(cusolverDnHandle_t handle,
    cublasSideMode_t side, cublasOperation_t trans, const int m, const int n,
    const int k, const float *A, const int lda, const float *tau, float *C,
    const int ldc, float *work, const int lwork, int *info)
  {
    return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
      work, lwork, info);
  }
  
  inline cusolverStatus_t ormqr(cusolverDnHandle_t handle,
    cublasSideMode_t side, cublasOperation_t trans, const int m, const int n,
    const int k, const double *A, const int lda, const double *tau, double *C,
    const int ldc, double *work, const int lwork, int *info)
  {
    return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
      work, lwork, info);
  }
  
  
  
  inline cusolverStatus_t orgqr_buflen(cusolverDnHandle_t handle, int m,
    int n, int k, const float *A, int lda, const float *tau, int *lwork)
  {
    return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
  }
  
  inline cusolverStatus_t orgqr_buflen(cusolverDnHandle_t handle, int m,
    int n, int k, const double *A, int lda, const double *tau, int *lwork)
  {
    return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
  }
  
  inline cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m,
    int n, int k, float *A, int lda, const float *tau, float *work, int lwork,
    int *info)
  {
    return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
  }
  
  inline cusolverStatus_t orgqr(cusolverDnHandle_t handle, int m,
    int n, int k, double *A, int lda, const double *tau, double *work, int lwork,
    int *info)
  {
    return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
  }
  
  
  
  inline cusolverStatus_t potrf_buflen(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, int n, float *A, int lda, int *lwork)
  {
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t potrf_buflen(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, int n, double *A, int lda, int *lwork)
  {
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t potrf(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, const int n, float *A, const int lda, float *work,
    const int lwork, int *info)
  {
    return cusolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, info);
  }
  
  inline cusolverStatus_t potrf(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, const int n, double *A, const int lda, double *work,
    const int lwork, int *info)
  {
    return cusolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, info);
  }
  
  
  
  inline cusolverStatus_t potri_buflen(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, int n, float *A, int lda, int *lwork)
  {
    return cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t potri_buflen(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, int n, double *A, int lda, int *lwork)
  {
    return cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork);
  }
  
  inline cusolverStatus_t potri(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, const int n, float *A, const int lda, float *work,
    const int lwork, int *info)
  {
    return cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, info);
  }
  
  inline cusolverStatus_t potri(cusolverDnHandle_t handle,
    cublasFillMode_t uplo, const int n, double *A, const int lda, double *work,
    const int lwork, int *info)
  {
    return cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, info);
  }
}
}


#endif
