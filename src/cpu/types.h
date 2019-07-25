#ifndef FML_MPI_TYPES_H
#define FML_MPI_TYPES_H


typedef int len_t;


typedef struct mat_t
{
  len_t m;
  len_t n;
  double *data;
} mat_t;


#endif
