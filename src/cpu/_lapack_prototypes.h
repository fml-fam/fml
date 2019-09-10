#ifndef FML_CPU__LAPACK_PROTOTYPES_H
#define FML_CPU__LAPACK_PROTOTYPES_H


#ifndef restrict
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif


extern void sgemm_(const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const float *restrict alpha,
  const float *restrict a, const int *lda, const float *restrict b,
  const int *ldb, const float *beta, float *restrict c, const int *ldc);

extern void dgemm_(const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const double *restrict alpha,
  const double *restrict a, const int *lda, const double *restrict b,
  const int *ldb, const double *beta, double *restrict c, const int *ldc);



extern void ssyrk_(const char *uplo, const char *trans, const int *n,
  const int *k, const float *restrict alpha, const float *restrict a,
  const int *lda, const float *restrict beta, float *restrict c,
  const int *ldc);

extern void dsyrk_(const char *uplo, const char *trans, const int *n,
  const int *k, const double *restrict alpha, const double *restrict a,
  const int *lda, const double *restrict beta, double *restrict c,
  const int *ldc);



extern void sgetrf_(const int *const restrict m, const int *const restrict n,
  float *const restrict a, const int *const lda,
  int *const restrict ipiv, int *const restrict info);

extern void dgetrf_(const int *const restrict m, const int *const restrict n,
  double *const restrict a, const int *const lda,
  int *const restrict ipiv, int *const restrict info);



extern void sgesdd_(const char *const restrict jobz, const int *const restrict m,
  const int *const restrict n, float *const restrict a,
  const int *const restrict lda, float *const restrict s,
  float *const restrict u, const int *const restrict ldu,
  float *const restrict vt, const int *const restrict ldvt,
  float *const restrict work, const int *const restrict lwork,
  int *const restrict iwork, int *const restrict info);

extern void dgesdd_(const char *const restrict jobz, const int *const restrict m,
  const int *const restrict n, double *const restrict a,
  const int *const restrict lda, double *const restrict s,
  double *const restrict u, const int *const restrict ldu,
  double *const restrict vt, const int *const restrict ldvt,
  double *const restrict work, const int *const restrict lwork,
  int *const restrict iwork, int *const restrict info);



extern void ssyevr_(const char *restrict jobz, const char *restrict range,
  const char *restrict uplo, const int *restrict n, float *restrict a,
  const int *restrict lda, const float *restrict vl, const float *restrict vu,
  const int *restrict il, const int *restrict iu, const float *restrict abstol,
  const int *restrict m, float *w, float *z, const int *restrict ldz,
  int *isuppz, float *work, const int *lwork, int *iwork, const int *liwork,
  int *info);

extern void dsyevr_(const char *restrict jobz, const char *restrict range,
  const char *restrict uplo, const int *restrict n, double *restrict a,
  const int *restrict lda, const double *restrict vl, const double *restrict vu,
  const int *restrict il, const int *restrict iu, const double *restrict abstol,
  const int *restrict m, double *w, double *z, const int *restrict ldz,
  int *isuppz, double *work, const int *lwork, int *iwork, const int *liwork,
  int *info);


#ifdef __cplusplus
}
#endif


#endif
