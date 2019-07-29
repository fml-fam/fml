#ifndef FML_MPI__SCALAPACK_PROTOTYPES_H
#define FML_MPI__SCALAPACK_PROTOTYPES_H


#ifdef __cplusplus
extern "C" {
#endif


void psgetrf_(const int *const restrict m, const int *const restrict n,
  float *const restrict a, const int *const restrict ia,
  const int *const restrict ja, const int *const restrict desca,
  int *const restrict ipiv, int *const restrict info);

void pdgetrf_(const int *const restrict m, const int *const restrict n,
  double *const restrict a, const int *const restrict ia,
  const int *const restrict ja, const int *const restrict desca,
  int *const restrict ipiv, int *const restrict info);


#ifdef __cplusplus
}
#endif


#endif
