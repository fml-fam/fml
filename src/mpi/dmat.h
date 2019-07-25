#ifndef FML_MPI_DMAT_H
#define FML_MPI_DMAT_H


#include <stdlib.h>
#include <string.h>

#include "../rand.h"
#include "utils.h"


static inline int fml_dmat_init(dmat_t *x, len_t m, len_t n, int mb, int nb, grid_t *g)
{
  len_local_t m_local = fml_numroc(m, mb, g->myrow, 0, g->nprow);
  len_local_t n_local = fml_numroc(n, nb, g->mycol, 0, g->npcol);
  
  double *data = malloc(m_local*n_local * sizeof(*data));
  if (data == NULL)
    return FML_EXIT_ERROR_MALLOC;
  
  descinit(x->desc, g->ictxt, m, n, mb, nb, m_local);
  
  x->m = m;
  x->n = n;
  x->m_local = m_local;
  x->n_local = n_local;
  x->mb = mb;
  x->nb = nb;
  x->data = data;
  x->g = g;
  
  #undef DATA
  #undef LOCM
  #undef LOCN
  
  return FML_EXIT_OK;
}



static inline void fml_dmat_free(dmat_t *x)
{
  free(x->data);
  x->data = NULL;
}



static inline void fml_dmat_fill_rand(dmat_t *x, int seed)
{
  fml_rand_init(seed);
  
  for (len_local_t j=0; j<(x->n_local); j++)
  {
    for (len_local_t i=0; i<(x->m_local); i++)
      (x->data)[i + (x->m_local)*j] = fml_rand_unif(0, 1);
  }
}



static inline void fml_dmat_fill_zero(dmat_t *x)
{
  double *data = x->data;
  memset(data, 0, (x->m_local)*(x->n_local) * sizeof(*data));
}



// DATA(y) := DATA(x)
static inline int fml_dmat_copy(dmat_t *x, dmat_t *y)
{
  if (x->m != y->m || x->n != y->n || x->mb != y->mb || x->nb != y->nb || x->desc[DESC_CTXT] != y->desc[DESC_CTXT] || x->g != y->g)
    return FML_EXIT_ERROR_NONCONFORMABLE;
  
  for (len_local_t j=0; j<(x->n_local); j++)
  {
    for (len_local_t i=0; i<(x->m_local); i++)
    {
      len_local_t ind = i + (x->m_local)*j;
      (x->data)[ind] = (y->data)[ind];
    }
  }
}


#endif
