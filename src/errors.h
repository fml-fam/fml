#ifndef FML_ERRORS_H
#define FML_ERRORS_H


#include <stdio.h>
#include <stdlib.h>

#define FML_EXIT_OK 0

#define FML_EXIT_ERROR_MALLOC -1
#define FML_ERROR_MALLOC_STRING "ERROR: malloc failed\n"

#define FML_EXIT_ERROR_NONCONFORMABLE -2
#define FML_ERROR_NONCONFORMABLE_STRING "ERROR: operation on non-conformable arrays"



static inline void fml_check_error(int code)
{
  if (code != FML_EXIT_OK)
  {
    if (code == FML_EXIT_ERROR_MALLOC)
      fprintf(stderr, FML_ERROR_MALLOC_STRING);
    else if (code == FML_EXIT_ERROR_NONCONFORMABLE)
      fprintf(stderr, FML_ERROR_NONCONFORMABLE_STRING);
    
    exit(code);
  }
}


#endif
