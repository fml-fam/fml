#ifndef FML_MPIMAT_SCALAPACK_H
#define FML_MPIMAT_SCALAPACK_H


#include "_scalapack_prototypes.h"


namespace scalapack
{
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
