// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_INTERNALS__SCALAPACK_PROTOTYPES_H
#define FML_MPI_INTERNALS__SCALAPACK_PROTOTYPES_H
#pragma once


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



extern void pssyevr_(const char *restrict jobz, const char *restrict range,
  const char *restrict uplo, const int *n, float *restrict a, const int *ia,
  const int *ja, const int *restrict desca, const float *vl, const float *vu,
  const int *il, const int *iu, int *m, int *nz, float *w, float *z,
  const int *iz, const int *jz, const int *restrict descz,
  float *restrict work, const int *restrict lwork, int *restrict iwork,
  const int *restrict liwork, int *restrict info);

extern void pdsyevr_(const char *restrict jobz, const char *restrict range,
  const char *restrict uplo, const int *n, double *restrict a, const int *ia,
  const int *ja, const int *restrict desca, const double *vl, const double *vu,
  const int *il, const int *iu, int *m, int *nz, double *w, double *z,
  const int *iz, const int *jz, const int *restrict descz,
  double *restrict work, const int *restrict lwork, int *restrict iwork,
  const int *restrict liwork, int *restrict info);



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



extern void psgetri_(const int *n, float *restrict a, const int *ia,
  const int *ja, const int *restrict desca, int *restrict ipiv,
  float *restrict work, const int *restrict lwork, int *restrict iwork,
  const int *restrict liwork, int *restrict info);

extern void pdgetri_(const int *n, double *restrict a, const int *ia,
  const int *ja, const int *restrict desca, int *restrict ipiv,
  double *restrict work, const int *restrict lwork, int *restrict iwork,
  const int *restrict liwork, int *restrict info);



extern void psgesv_(const int *n, const int *nrhs, float *restrict a,
  const int *ia, const int *ja, const int *restrict desca, int *restrict ipvt,
  float *restrict b, const int *ib, const int *jb, const int *restrict descb,
  int *info);

extern void pdgesv_(const int *n, const int *nrhs, double *restrict a,
  const int *ia, const int *ja, const int *restrict desca, int *restrict ipvt,
  double *restrict b, const int *ib, const int *jb, const int *restrict descb,
  int *info);



extern void pslacpy_(const char *uplo, const int *m, const int *n,
  const float *restrict a, const int *ia, const int *ja, const int *desca,
  float *restrict b, const int *ib, const int *jb, const int *descb);

extern void pdlacpy_(const char *uplo, const int *m, const int *n,
  const double *restrict a, const int *ia, const int *ja, const int *desca,
  double *restrict b, const int *ib, const int *jb, const int *descb);



extern void psgeqpf_(const int *m, const int *n, float *restrict a,
  const int *ia, const int *ja, const int *desca, int *restrict ipiv,
  float *restrict tau, float *restrict work, const int *lwork,
  int *info);

extern void pdgeqpf_(const int *m, const int *n, double *restrict a,
  const int *ia, const int *ja, const int *desca, int *restrict ipiv,
  double *restrict tau, double *restrict work, const int *lwork,
  int *info);



extern void psgeqrf_(const int *m, const int *n, float *restrict a,
  const int *ia, const int *ja, const int *desca, float *restrict tau,
  float *restrict work, const int *lwork, int *info);

extern void pdgeqrf_(const int *m, const int *n, double *restrict a,
  const int *ia, const int *ja, const int *desca, double *restrict tau,
  double *restrict work, const int *lwork, int *info);



extern void psormqr_(const char *side, const char *trans, const int *m,
  const int *n, const int *k, const float *restrict a, const int *ia,
  const int *ja, const int *desca, float *restrict tau, float *restrict c,
  const int *ic, const int *jc, const int *descc, float *restrict work,
  const int *lwork, int *info);

extern void pdormqr_(const char *side, const char *trans, const int *m,
  const int *n, const int *k, const double *restrict a, const int *ia,
  const int *ja, const int *desca, double *restrict tau, double *restrict c,
  const int *ic, const int *jc, const int *descc, double *restrict work,
  const int *lwork, int *info);


#ifdef __cplusplus
}
#endif


#endif
