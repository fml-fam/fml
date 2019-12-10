// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML__INTERNALS_UNIVEC_H
#define FML__INTERNALS_UNIVEC_H
#pragma once


#include <stdexcept>

#include "types.hh"


/**
 * @brief Base vector class. Not meant for direct use. Instead see `cpuvec`
 * and `gpuvec`.
 */
template <typename T>
class univec
{
  public:
    /// Number of elements in the vector.
    len_t size() const {return _size;};
    /// Pointer to the internal array.
    T* data_ptr() {return data;};
    T* data_ptr() const {return data;};
  
  protected:
    len_t _size;
    T *data;
    bool free_data;
    
    bool should_free() const {return free_data;};
    void check_index(const len_t i) const;
    void printval(const T val, uint8_t ndigits) const;
};



template <typename REAL>
void univec<REAL>::check_index(const len_t i) const
{
  if (i < 0 || i >= this->_size)
    throw std::runtime_error("index out of bounds");
}



template <>
inline void univec<int>::printval(const int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

template <typename T>
void univec<T>::printval(const T val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}


#endif
