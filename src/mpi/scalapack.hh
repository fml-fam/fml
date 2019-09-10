#ifndef FML_MPI_SCALAPACK_H
#define FML_MPI_SCALAPACK_H


#include "_scalapack_prototypes.h"


namespace scalapack
{
  inline void gemm(const char transa, const char transb, const int m, const int n,
    const int k, const float alpha, const float *a, const int *desca,
    const float *b, const int *descb, const float beta, float *c,
    const int *descc)
  {
    int ij = 1;
    psgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &ij, &ij, desca, b,
      &ij, &ij, descb, &beta, c, &ij, &ij, descc);
  }
  
  inline void gemm(const char transa, const char transb, const int m, const int n,
    const int k, const double alpha, const double *a, const int *desca,
    const double *b, const int *descb, const double beta, double *c,
    const int *descc)
  {
    int ij = 1;
    pdgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &ij, &ij, desca, b,
      &ij, &ij, descb, &beta, c, &ij, &ij, descc);
  }
  
  
  
  inline void syrk(const char uplo, const char trans, const int n, const int k, 
    const float alpha, const float *a, const int *desca,
    const float beta, float *c, const int *descc)
  {
    int ij = 1;
    pssyrk_(&uplo, &trans, &n, &k, &alpha, a, &ij, &ij, desca, &beta, c, &ij,
      &ij, descc);
  }
  
  inline void syrk(const char uplo, const char trans, const int n, const int k, 
    const double alpha, const double *a, const int *desca,
    const double beta, double *c, const int *descc)
  {
    int ij = 1;
    pdsyrk_(&uplo, &trans, &n, &k, &alpha, a, &ij, &ij, desca, &beta, c, &ij,
      &ij, descc);
  }
  
  
  
  inline void tran(const int m, const int n, const float alpha, const float *a,
    const int *desca, const float beta, float *c, const int *descc)
  {
    int ij = 1;
    pstran_(&m, &n, &alpha, a, &ij, &ij, desca, &beta, c, &ij, &ij, descc);
  }
  
  inline void tran(const int m, const int n, const double alpha, const double *a,
    const int *desca, const double beta, double *c, const int *descc)
  {
    int ij = 1;
    pdtran_(&m, &n, &alpha, a, &ij, &ij, desca, &beta, c, &ij, &ij, descc);
  }
  
  
  
  inline void getrf(const int m, const int n, float *a, int *desca, int *ipiv,
    int *info)
  {
    int ij = 1;
    psgetrf_(&m, &n, a, &ij, &ij, desca, ipiv, info);
  }
  
  inline void getrf(const int m, const int n, double *a, int *desca, int *ipiv,
    int *info)
  {
    int ij = 1;
    pdgetrf_(&m, &n, a, &ij, &ij, desca, ipiv, info);
  }
  
  
  
  inline void gesvd(const char jobu, const char jobvt, const int m, const int n,
    float *a, int *desca, float *s, float *u, int *descu, float *vt,
    int *descvt, float *work, int lwork, int *info)
  {
    int ij = 1;
    psgesvd_(&jobu, &jobvt, &m, &n, a, &ij, &ij, desca, s, u, &ij, &ij, descu,
      vt, &ij, &ij, descvt, work, &lwork, info);
  }
  
  inline void gesvd(const char jobu, const char jobvt, const int m, const int n,
    double *a, int *desca, double *s, double *u, int *descu, double *vt,
    int *descvt, double *work, int lwork, int *info)
  {
    int ij = 1;
    pdgesvd_(&jobu, &jobvt, &m, &n, a, &ij, &ij, desca, s, u, &ij, &ij, descu, vt, &ij, &ij, descvt, work, &lwork, info);
  }
}


#endif
