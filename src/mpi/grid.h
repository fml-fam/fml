#ifndef FML_MPI_GRID_H
#define FML_MPI_GRID_H


#include <math.h>
#include <mpi.h>

#include "_blacs_prototypes.h"
#include "errors.h"

#define PROC_GRID_SQUARE  0
#define PROC_GRID_WIDE    1
#define PROC_GRID_TALL    2


static inline int fml_grid_init(grid_t *g, int gridtype)
{
  char order = 'R';
  int size;
  int ictxt;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  Cblacs_get(-1, 0, &ictxt);
  
  if (gridtype == PROC_GRID_SQUARE)
  {
    int nr, nc;
    int n = (int) sqrt((double) size);
    n = (n<1)?1:n; // suppresses bogus compiler warning
    
    for (int i=0; i<n; i++)
    {
      nc = n - i;
      nr = size % nc;
      if (nr == 0)
        break;
    }
    
    nr = size / nc;
    
    Cblacs_gridinit(&ictxt, &order, nr, nc);
  }
  else if (gridtype == PROC_GRID_TALL)
    Cblacs_gridinit(&ictxt, &order, size, 1);
  else if (gridtype == PROC_GRID_WIDE)
    Cblacs_gridinit(&ictxt, &order, 1, size);
  else
    return FML_EXIT_ERROR_BLACSGRID;
  
  Cblacs_gridinfo(ictxt, &(g->nprow), &(g->npcol), &(g->myrow), &(g->mycol));
  g->ictxt = ictxt;
  g->size = size;
  
  return FML_EXIT_OK;
}



static inline void fml_grid_set(grid_t *g, int ictxt)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  g->size = size;
  g->ictxt = ictxt;
  
  Cblacs_gridinfo(ictxt, &(g->nprow), &(g->npcol), &(g->myrow), &(g->mycol));
}



static inline void fml_grid_finalize(grid_t *g)
{
  int mpi_continue = 0;
  Cblacs_gridexit(g->ictxt);
  Cblacs_exit(mpi_continue);
}


#endif
