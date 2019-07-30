#ifndef FML_GPUMAT_CUSOLVER_H
#define FML_GPUMAT_CUSOLVER_H


#include <cusolverDn.h>


namespace lapack
{
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
