#ifndef FML_MPI_TYPES_H
#define FML_MPI_TYPES_H


#include "../types.h"
typedef int len_local_t;


enum desc_pos
{
  DESC_DTYPE,
  DESC_CTXT,
  DESC_M,
  DESC_N,
  DESC_MB,
  DESC_NB,
  DESC_RSRC,
  DESC_CSRC,
  DESC_LLD
};



enum blacs_pos
{
  GRID_NPROCS,
  GRID_NPROW,
  GRID_NPCOL,
  GRID_MYPROW,
  GRID_MYPCOL
};



typedef struct grid_t
{
  int size;
  int ictxt;
  int nprow;
  int npcol;
  int myrow;
  int mycol;
} grid_t;



typedef struct dmat_t
{
  len_t m;
  len_t n;
  len_local_t m_local;
  len_local_t n_local;
  int mb;
  int nb;
  double *data;
  int desc[9];
  grid_t *g;
} dmat_t;


#endif
