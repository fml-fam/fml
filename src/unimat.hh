#ifndef FML_MATRIX_H
#define FML_MATRIX_H


#ifdef __CUDACC__
#include <cublas.h>
#endif

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
    void printval(const REAL val, uint8_t ndigits) const;
};



template <>
inline void unimat<int>::printval(const int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

#ifdef __CUDACC__
template <>
inline void unimat<__half>::printval(const __half val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, (float)val);
}
#endif

template <typename REAL>
void unimat<REAL>::printval(const REAL val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}


#endif
