#ifndef FML_MPI_SCALAPACK_H
#define FML_MPI_SCALAPACK_H


#include "_scalapack_prototypes.h"


namespace scalapack
{
  void gemm(const char transa, const char transb, const int m, const int n,
    const int k, const float alpha, const float *a, const int *desca,
    const float *b, const int *descb, const float beta, float *c,
    const int *descc)
  {
    int ij = 1;
    psgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &ij, &ij, desca, b,
      &ij, &ij, descb, &beta, c, &ij, &ij, descc);
  }
  
  void gemm(const char transa, const char transb, const int m, const int n,
    const int k, const double alpha, const double *a, const int *desca,
    const double *b, const int *descb, const double beta, double *c,
    const int *descc)
  {
    int ij = 1;
    pdgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &ij, &ij, desca, b,
      &ij, &ij, descb, &beta, c, &ij, &ij, descc);
  }
  
  
  
  void syrk(const char uplo, const char trans, const int n, const int k, 
    const float alpha, const float *a, const int *desca,
    const float beta, float *restrict c, const int *descc)
  {
    int ij = 1;
    pssyrk_(&uplo, &trans, &n, &k, &alpha, a, &ij, &ij, desca, &beta, c, &ij,
      &ij, descc);
  }
  
  void syrk(const char uplo, const char trans, const int n, const int k, 
    const double alpha, const double *a, const int *desca,
    const double beta, double *restrict c, const int *descc)
  {
    int ij = 1;
    pdsyrk_(&uplo, &trans, &n, &k, &alpha, a, &ij, &ij, desca, &beta, c, &ij,
      &ij, descc);
  }
  
  
  
  void getrf(const int m, const int n, float *a, int *desca, int *ipiv, int *info)
  {
    int ij = 1;
    psgetrf_(&m, &n, a, &ij, &ij, desca, ipiv, info);
  }
  
  void getrf(const int m, const int n, double *a, int *desca, int *ipiv, int *info)
  {
    int ij = 1;
    pdgetrf_(&m, &n, a, &ij, &ij, desca, ipiv, info);
  }
}


#endif
