#ifndef FML_LINALGUTILS_H
#define FML_LINALGUTILS_H


#include "types.hh"


namespace linalgutils
{
  inline void matmult_params(const bool transx, const bool transy, const len_t mx, const len_t nx, const len_t my, const len_t ny, len_t *m, len_t *n, len_t *k)
  {
    // m = # rows of op(x)
    // n = # cols of op(y)
    // k = # cols of op(x)
    
    if (transx)
    {
      *m = nx;
      *k = mx;
    }
    else
    {
      *m = mx;
      *k = nx;
    }
    
    *n = transy ? my : ny;
  }
}


#endif
