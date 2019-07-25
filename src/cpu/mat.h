#ifndef FML_CPU_MAT_H
#define FML_CPU_MAT_H


#include <stdlib.h>
#include <string.h>

#include "../errors.h"
#include "../rand.h"

#include "types.h"


static inline int fml_mat_init(mat_t *x, len_t m, len_t n)
{
  double *data = malloc(m*n * sizeof(*data));
  if (data == NULL)
    return FML_EXIT_ERROR_MALLOC;
  
  x->m = m;
  x->n = n;
  x->data = data;
  
  return FML_EXIT_OK;
}



static inline void fml_dmat_free(mat_t *x)
{
  free(x->data);
  x->data = NULL;
}



static inline void fml_dmat_fill_rand(mat_t *x, int seed)
{
  fml_rand_init(seed);
  
  for (int j=0; j<(x->n); j++)
  {
    for (int i=0; i<(x->m); i++)
      (x->data)[i + (x->m)*j] = fml_rand_unif(0, 1);
  }
}



static inline void fml_dmat_fill_zero(mat_t *x)
{
  double *data = x->data;
  memset(data, 0, (x->m)*(x->n) * sizeof(*data));
}



// DATA(y) := DATA(x)
static inline int fml_dmat_copy(mat_t *x, mat_t *y)
{
  if (x->m != y->m || x->n != y->n)
    return FML_EXIT_ERROR_NONCONFORMABLE;
  
  for (len_t j=0; j<(x->n); j++)
  {
    for (len_t i=0; i<(x->m); i++)
    {
      len_t ind = i + (x->m)*j;
      (x->data)[ind] = (y->data)[ind];
    }
  }
  
  
  return FML_EXIT_OK;
}


#endif
