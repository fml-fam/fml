#ifndef FML_CPU_LAPACK_H
#define FML_CPU_LAPACK_H


#include "_lapack_prototypes.h"


namespace lapack
{
  void gemm(const char transa, const char transb, const int m,
    const int n, const int k, const float alpha,
    const float *restrict a, const int lda, const float *restrict b,
    const int ldb, const float beta, float *restrict c, const int ldc)
  {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  
  void gemm(const char transa, const char transb, const int m,
    const int n, const int k, const double alpha,
    const double *restrict a, const int lda, const double *restrict b,
    const int ldb, const double beta, double *restrict c, const int ldc)
  {
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  
  
  
  void syrk(const char uplo, const char trans, const int n, const int k, 
    const float alpha, const float *restrict a, const int lda,
    const float beta, float *restrict c, const int ldc)
  {
    ssyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
  }
  
  void syrk(const char uplo, const char trans, const int n, const int k, 
    const double alpha, const double *restrict a, const int lda,
    const double beta, double *restrict c, const int ldc)
  {
    dsyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
  }
  
  
  
  void getrf(const int m, const int n, float *a, const int lda, int *ipiv, int *info)
  {
    sgetrf_(&m, &n, a, &lda, ipiv, info);
  }
  
  void getrf(const int m, const int n, double *a, const int lda, int *ipiv, int *info)
  {
    dgetrf_(&m, &n, a, &lda, ipiv, info);
  }
}


#endif
