#ifndef FML_CPUMAT_LAPACK_H
#define FML_CPUMAT_LAPACK_H


#include "_lapack_prototypes.h"


namespace lapack
{
  void getrf(const int m, const int n, float *a, int *ipiv, int *info)
  {
    sgetrf_(&m, &n, a, &m, ipiv, info);
  }
  
  void getrf(const int m, const int n, double *a, int *ipiv, int *info)
  {
    dgetrf_(&m, &n, a, &m, ipiv, info);
  }
}


#endif
