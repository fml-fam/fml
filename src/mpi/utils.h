#ifndef FML_MPI_UTILS_H
#define FML_MPI_UTILS_H


#include <math.h>
#include <stdlib.h>

#include "_blacs_prototypes.h"
#include "errors.h"


static inline int fml_numroc(int n, int nb, int iproc, int isrcproc, int nprocs)
{
  int mydist = (nprocs+iproc-isrcproc) % nprocs;
  int nblocks = n / nb;

  int ret = (nblocks/nprocs) * nb;

  int extrablks = nblocks % nprocs;
  if (mydist < extrablks)
    ret += nb;
  else if (mydist == extrablks)
    ret += (n%nb);

  return ret;
}



static inline void fml_pdims(const int *const restrict desc, int *const restrict ldm, int *const restrict blacs)
{
  Cblacs_gridinfo(desc[DESC_CTXT], blacs+GRID_NPROW, blacs+GRID_NPCOL, blacs+GRID_MYPROW, blacs+GRID_MYPCOL);
  
  if (blacs[GRID_NPROW] == -1 || blacs[GRID_NPCOL] == -1)
    blacs[GRID_NPROCS] = -1;
  else
    blacs[GRID_NPROCS] = blacs[GRID_NPROW] * blacs[GRID_NPCOL];
  
  ldm[0] = fml_numroc(desc[DESC_M], desc[DESC_MB], blacs[GRID_MYPROW], desc[DESC_RSRC], blacs[GRID_NPROW]);
  ldm[1] = fml_numroc(desc[DESC_N], desc[DESC_NB], blacs[GRID_MYPCOL], desc[DESC_CSRC], blacs[GRID_NPCOL]);
  
  if (ldm[0] < 1 || ldm[1] < 1)
  {
    ldm[0] = 0;
    ldm[1] = 0;
  }
}



static inline void fml_l2gpair(const int i, const int j, int *const restrict gi, int *const restrict gj, const int *const restrict desc, const int *const restrict blacs)
{
  const int nprocs = blacs[GRID_NPROCS];
  const int mb = desc[DESC_MB];
  const int nb = desc[DESC_NB];
  
  *gi = nprocs*mb * (i-1)/mb + (i-1)%mb + ((nprocs+blacs[GRID_MYPROW])%nprocs)*mb + 1;
  *gj = nprocs*nb * (j-1)/nb + (j-1)%nb + ((nprocs+blacs[GRID_MYPCOL])%nprocs)*nb + 1;
}


#endif
