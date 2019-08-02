#ifndef FML_LINALGUTILS_H
#define FML_LINALGUTILS_H


namespace linalgutils
{
  void matmult_params(const bool transx, const bool transy, const int mx, const int nx, const int my, const int ny, int *m, int *n, int *k)
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
