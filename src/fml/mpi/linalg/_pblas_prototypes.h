// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG__PBLAS_PROTOTYPES_H
#define FML_MPI_LINALG__PBLAS_PROTOTYPES_H
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
