// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MATRIX_H
#define FML_MATRIX_H
#pragma once


#ifdef __CUDACC__
#include <cublas.h>
#endif

#include <stdexcept>

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
    void check_index(const len_t i) const;
    void check_index(const len_t i, const len_t j) const;
    void printval(const REAL val, uint8_t ndigits) const;
};



template <typename REAL>
void unimat<REAL>::check_index(const len_t i) const
{
  if (i < 0 || i >= (this->m * this->n))
    throw std::runtime_error("index out of bounds");
}

template <typename REAL>
void unimat<REAL>::check_index(const len_t i, const len_t j) const
{
  if (i < 0 || i >= this->m || j < 0 || j >= this->n)
    throw std::runtime_error("index out of bounds");
}



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
