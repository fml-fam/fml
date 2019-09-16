#ifndef FML_CPU_LAPACK_H
#define FML_CPU_LAPACK_H


#include "_lapack_prototypes.h"


namespace lapack
{
  inline void gemm(const char transa, const char transb, const int m,
    const int n, const int k, const float alpha,
    const float *restrict a, const int lda, const float *restrict b,
    const int ldb, const float beta, float *restrict c, const int ldc)
  {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
      &ldc);
  }
  
  inline void gemm(const char transa, const char transb, const int m,
    const int n, const int k, const double alpha,
    const double *restrict a, const int lda, const double *restrict b,
    const int ldb, const double beta, double *restrict c, const int ldc)
  {
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
      &ldc);
  }
  
  
  
  inline void syrk(const char uplo, const char trans, const int n, const int k, 
    const float alpha, const float *restrict a, const int lda,
    const float beta, float *restrict c, const int ldc)
  {
    ssyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
  }
  
  inline void syrk(const char uplo, const char trans, const int n, const int k, 
    const double alpha, const double *restrict a, const int lda,
    const double beta, double *restrict c, const int ldc)
  {
    dsyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
  }
  
  
  
  inline void getrf(const int m, const int n, float *a, const int lda,
    int *ipiv, int *info)
  {
    sgetrf_(&m, &n, a, &lda, ipiv, info);
  }
  
  inline void getrf(const int m, const int n, double *a, const int lda,
    int *ipiv, int *info)
  {
    dgetrf_(&m, &n, a, &lda, ipiv, info);
  }
  
  
  
  inline void gesdd(const char jobz, const int m, const int n, float *a,
    const int lda, float *s, float *u, const int ldu, float *vt,
    const int ldvt, float *work, const int lwork, int *iwork, int *info)
  {
    sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork,
      info);
  }
  
  inline void gesdd(const char jobz, const int m, const int n, double *a,
    const int lda, double *s, double *u, const int ldu, double *vt,
    const int ldvt, double *work, const int lwork, int *iwork, int *info)
  {
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork,
      info);
  }
  
  
  
  inline void syevr(const char jobz, const char range, const char uplo,
    const int n, float *a, const int lda, const float vl, const float vu,
    const int il, const int iu, const float abstol, int *m, float *w,
    float *z, const int ldz, int *isuppz, float *work, const int lwork,
    int *iwork, const int liwork, int *info)
  {
    ssyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m,
      w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, info);
  }
  
  inline void syevr(const char jobz, const char range, const char uplo,
    const int n, double *a, const int lda, const double vl, const double vu,
    const int il, const int iu, const double abstol, int *m, double *w,
    double *z, const int ldz, int *isuppz, double *work, const int lwork,
    int *iwork, const int liwork, int *info)
  {
    dsyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m,
      w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, info);
  }
  
  
  
  inline void geev(const char jobvl, const char jobvr, const int n, float *a,
    const int lda, float *wr, float *wi, float *vl, const int ldvl, float *vr,
    const int ldvr, float *work, const int lwork, int *info)
    {
      sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
        &lwork, info);
    }
  
  inline void geev(const char jobvl, const char jobvr, const int n, double *a,
    const int lda, double *wr, double *wi, double *vl, const int ldvl, double *vr,
    const int ldvr, double *work, const int lwork, int *info)
    {
      dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
        &lwork, info);
    }
  
  
  
  inline void getri(const int n, float *a, const int lda, int *ipiv,
    float *work, const int lwork, int *info)
  {
    sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
  }
  
  inline void getri(const int n, double *a, const int lda, int *ipiv,
    double *work, const int lwork, int *info)
  {
    dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
  }
  
  
  
  inline void gesv(const int n, const int nrhs, float *a, const int lda,
    int *ipiv, float *b, const int ldb, int *info)
  {
    sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
  }
  
  inline void gesv(const int n, const int nrhs, double *a, const int lda,
    int *ipiv, double *b, const int ldb, int *info)
  {
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
  }
}


#endif
