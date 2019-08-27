#ifndef FML_MATRIX_H
#define FML_MATRIX_H


#include "types.hh"


template <typename REAL>
class matrix
{
  public:
    len_t nrows() const {return m;};
    len_t ncols() const {return n;};
    REAL* data_ptr() {return data;};
    REAL* data_ptr() const {return data;};
  
  protected:
    len_t m;
    len_t n;
    REAL *data;
};


#endif
