#ifndef FML_MPI__SCALAPACK_PROTOTYPES_H
#define FML_MPI__SCALAPACK_PROTOTYPES_H


#ifdef __cplusplus
extern "C" {
#endif


void psgemm_(const char *transa, const char *transb, const int *m, const int *n,
  const int *k, const float *alpha, const float *a, const int *ia,
  const int *ja, const int *desca, const float *b, const int *ib,
  const int *jb, const int *descb, const float *beta, float *c, const int *ic,
  const int *jc, const int *descc);

void pdgemm_(const char *transa, const char *transb, const int *m, const int *n,
  const int *k, const double *alpha, const double *a, const int *ia,
  const int *ja, const int *desca, const double *b, const int *ib,
  const int *jb, const int *descb, const double *beta, double *c, const int *ic,
  const int *jc, const int *descc);



extern void psgetrf_(const int *m, const int *n, float *a, const int *ia,
  const int *ja, const int *desca, int *ipiv, int *info);

extern void pdgetrf_(const int *m, const int *n, double *a, const int *ia,
  const int *ja, const int *desca, int *ipiv, int *info);


#ifdef __cplusplus
}
#endif


#endif
