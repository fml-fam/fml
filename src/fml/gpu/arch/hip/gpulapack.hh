// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_HIP_CUSOLVER_H
#define FML_GPU_ARCH_HIP_CUSOLVER_H
#pragma once


#include <rocblas.h>
#include <rocsolver.h>


namespace gpulapack
{
  namespace err
  {
    inline std::string get_rocsolver_error_msg(cusolverStatus_t check)
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
    
    inline void check_gpusolver_ret(rocsolver_status check, std::string op)
    {
      if (check != CUSOLVER_STATUS_SUCCESS)
      {
        std::string msg = "rocsolver " + op + "() failed with error: " + get_rocsolver_error_msg(check);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  inline rocsolver_status getrf_buflen(rocsolver_handle handle, int m, int n,
     float *A, int lda, int *lwork)
  {
    return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  inline rocsolver_status getrf_buflen(rocsolver_handle handle, int m, int n,
     double *A, int lda, int *lwork)
  {
    return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork);
  }
  
  inline rocsolver_status getrf(rocsolver_handle handle, int m, int n,
     float *A, int lda, float *work, int *ipiv, int *info)
  {
    return cusolverDnSgetrf(handle, m, n, A, lda, work, ipiv, info);
  }
  
  inline rocsolver_status getrf(rocsolver_handle handle, int m, int n,
     double *A, int lda, double *work, int *ipiv, int *info)
  {
    return cusolverDnDgetrf(handle, m, n, A, lda, work, ipiv, info);
  }
  
  
  
  inline rocsolver_status gesvd_buflen(rocsolver_handle handle, int m, int n,
     float *A, int *lwork)
  {
    (void)A;
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
  }
  
  inline rocsolver_status gesvd_buflen(rocsolver_handle handle, int m, int n,
     double *A, int *lwork)
  {
    (void)A;
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
  }
  
  inline rocsolver_status gesvd(rocsolver_handle handle, signed char jobu,
    signed char jobvt, const int m, const int n, float *A, const int lda,
    float *S, float *U, const int ldu, float *VT, const int ldvt, float *work,
    const int lwork, float *rwork, int *info)
  {
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
      ldvt, work, lwork, rwork, info);
  }
  
  inline rocsolver_status gesvd(rocsolver_handle handle, signed char jobu,
    signed char jobvt, const int m, const int n, double *A, const int lda,
    double *S, double *U, const int ldu, double *VT, const int ldvt, double *work,
    const int lwork, double *rwork, int *info)
  {
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
      ldvt, work, lwork, rwork, info);
  }
  
  
  
  inline rocsolver_status syevd_buflen(rocsolver_handle handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A,
    int lda, const float *W, int *lwork)
  {
    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W,
      lwork);
  }
  
  inline rocsolver_status syevd_buflen(rocsolver_handle handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A,
    int lda, const double *W, int *lwork)
  {
    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W,
      lwork);
  }
  
  inline rocsolver_status syevd(rocsolver_handle handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda,
    float *W, float *work, int lwork, int *devInfo)
  {
    return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
      devInfo);
  }
  
  inline rocsolver_status syevd(rocsolver_handle handle,
    cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda,
    double *W, double *work, int lwork, int *devInfo)
  {
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
      devInfo);
  }
  
  
  
  inline rocblas_status getri_batched(rocblas_handle handle, const int n,
    const float **Aarray, const int lda, const int *devIpiv, float **Carray,
    const int ldb, int *info, const int batchSize)
  {
    return cublasSgetriBatched(handle, n, Aarray, lda, devIpiv, Carray, ldb,
      info, batchSize);
  }
  
  inline rocblas_status getri_batched(rocblas_handle handle, const int n,
    const double **Aarray, const int lda, const int *devIpiv, double **Carray,
    const int ldb, int *info, const int batchSize)
  {
    return cublasDgetriBatched(handle, n, Aarray, lda, devIpiv, Carray, ldb,
      info, batchSize);
  }
  
  
  
  inline rocsolver_status getrs(rocsolver_handle handle,
    rocblas_operation trans, const int n, const int nrhs, const float *A,
    const int lda, const int *devIpiv, float *B, const int ldb, int *info)
  {
    return rocsolver_sgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
      info);
  }
  
  inline rocsolver_status getrs(rocsolver_handle handle,
    rocblas_operation trans, const int n, const int nrhs, const double *A,
    const int lda, const int *devIpiv, double *B, const int ldb, int *info)
  {
    return rocsolver_dgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb,
      info);
  }
}


#endif
