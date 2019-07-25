#ifndef FML_MPI_ERRORS_H
#define FML_MPI_ERRORS_H


#include "../errors.h"
#include "types.h"

#define FML_EXIT_ERROR_SCALAPACK -101
#define FML_ERROR_SCALAPACK_STRING "ERROR: ScaLAPACK function %s failed with info=%d"

#define FML_EXIT_ERROR_BLACSGRID -102
#define FML_ERROR_BLACSGRID_STRING "ERROR: impossible grid type"



static inline void fml_parcheck_error(grid_t *g, int code)
{
  if (code != FML_EXIT_OK)
  {
    if (g->myrow == 0 && g->mycol == 0)
    {
      if (code == FML_EXIT_ERROR_MALLOC)
        fprintf(stderr, FML_ERROR_MALLOC_STRING);
      if (code == FML_EXIT_ERROR_BLACSGRID)
        fprintf(stderr, FML_ERROR_BLACSGRID_STRING);
    }
    
    exit(code);
  }
}


#endif
