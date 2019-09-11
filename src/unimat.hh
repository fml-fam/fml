#ifndef FML_MATRIX_H
#define FML_MATRIX_H


#include "types.hh"


template <typename REAL>
class unimat
{
  public:
    bool is_square() const {return (this->m==this->n);};
    
    len_t nrows() const {return m;};
    len_t ncols() const {return n;};
    REAL* data_ptr() {return data;};
    REAL* data_ptr() const {return data;};
  
  protected:
    len_t m;
    len_t n;
    REAL *data;
    bool free_data;
    bool should_free() const {return free_data;};
};


#endif
