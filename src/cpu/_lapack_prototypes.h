#ifndef FML_CPU__LAPACK_PROTOTYPES_H
#define FML_CPU__LAPACK_PROTOTYPES_H


#ifdef __cplusplus
extern "C" {
#endif


extern void sgetrf_(const int *const restrict m, const int *const restrict n,
  float *const restrict a, const int *const lda,
  int *const restrict ipiv, int *const restrict info);

extern void dgetrf_(const int *const restrict m, const int *const restrict n,
  double *const restrict a, const int *const lda,
  int *const restrict ipiv, int *const restrict info);


#ifdef __cplusplus
}
#endif


#endif
