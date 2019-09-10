#ifndef FML_MPI__SCALAPACK_PROTOTYPES_H
#define FML_MPI__SCALAPACK_PROTOTYPES_H


#ifndef restrict
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif


extern void psgemm_(const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const float *restrict alpha,
  const float *restrict a, const int *ia, const int *ja, const int *desca,
  const float *restrict b, const int *ib, const int *jb, const int *descb,
  const float *restrict beta, float *restrict c, const int *ic, const int *jc,
  const int *descc);

extern void pdgemm_(const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const double *restrict alpha,
  const double *restrict a, const int *ia, const int *ja, const int *desca,
  const double *restrict b, const int *ib, const int *jb, const int *descb,
  const double *restrict beta, double *restrict c, const int *ic, const int *jc,
  const int *descc);



extern void pssyrk_(const char *uplo, const char *trans, const int *n,
  const int *k, const float *restrict alpha, const float *restrict a,
  const int *ia, const int *ja, const int *desca, const float *restrict beta,
  float *restrict c, const int *ic, const int *jc, const int *descc);
  
extern void pdsyrk_(const char *uplo, const char *trans, const int *n,
  const int *k, const double *restrict alpha, const double *restrict a,
  const int *ia, const int *ja, const int *desca, const double *restrict beta,
  double *restrict c, const int *ic, const int *jc, const int *descc);



extern void pstran_(const int *restrict m, const int *restrict n,
  const float *restrict alpha, const float *restrict a, const int *ia,
  const int *ja, const int *restrict desca, const float *restrict beta,
  float *restrict c, const int *ic, const int *jc, const int *restrict descc);

extern void pdtran_(const int *restrict m, const int *restrict n,
  const double *restrict alpha, const double *restrict a, const int *ia,
  const int *ja, const int *restrict desca, const double *restrict beta,
  double *restrict c, const int *ic, const int *jc, const int *restrict descc);



extern void psgetrf_(const int *m, const int *n, float *restrict a,
  const int *ia, const int *ja, const int *desca, int *restrict ipiv,
  int *info);

extern void pdgetrf_(const int *m, const int *n, double *restrict a,
  const int *ia, const int *ja, const int *desca, int *restrict ipiv,
  int *info);



extern void psgesvd_(const char *restrict jobu, const char *restrict jobvt,
  const int *restrict m, const int *restrict n, float *restrict a,
  const int *ia, const int *ja, const int *restrict desca,
  float *restrict s, float *restrict u, const int *restrict iu,
  const int *restrict ju, const int *restrict descu, float *restrict vt,
  const int *restrict ivt, const int *restrict jvt, const int *restrict descvt,
  float *restrict work, const int *restrict lwork, int *restrict info);

extern void pdgesvd_(const char *restrict jobu, const char *restrict jobvt,
  const int *restrict m, const int *restrict n, double *restrict a,
  const int *ia, const int *ja, const int *restrict desca,
  double *restrict s, double *restrict u, const int *restrict iu,
  const int *restrict ju, const int *restrict descu, double *restrict vt,
  const int *restrict ivt, const int *restrict jvt, const int *restrict descvt,
  double *restrict work, const int *restrict lwork, int *restrict info);



extern void psgeadd_(const char *restrict trans, const int *restrict m,
  const int *restrict n, const float *restrict alpha, const float *restrict a,
  const int *ia, const int *ja, const int *restrict desca,
  const float *restrict beta, float *restrict c, const int *ic, const int *jc,
  const int *restrict descc);

extern void pdgeadd_(const char *restrict trans, const int *restrict m,
  const int *restrict n, const double *restrict alpha, const double *restrict a,
  const int *ia, const int *ja, const int *restrict desca,
  const double *restrict beta, double *restrict c, const int *ic, const int *jc,
  const int *restrict descc);


#ifdef __cplusplus
}
#endif


#endif
