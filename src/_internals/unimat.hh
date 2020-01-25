// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML__INTERNALS_MATRIX_H
#define FML__INTERNALS_MATRIX_H
#pragma once


#ifdef __CUDACC__
#include <cublas.h>
#endif

#include <stdexcept>

#include "print.hh"
#include "types.hh"


/**
 * @brief Base matrix class. Not meant for direct use. Instead see `cpumat`,
 * `gpumat`, and `mpimat`.
 */
template <typename REAL>
class unimat
{
  public:
    /// Is the matrix square?
    bool is_square() const {return (this->m==this->n);};
    /// Number of rows.
    len_t nrows() const {return m;};
    /// Number of columns.
    len_t ncols() const {return n;};
    /// Pointer to the internal array.
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
  fml::print::printf("%d ", val);
}

#ifdef __CUDACC__
template <>
inline void unimat<__half>::printval(const __half val, uint8_t ndigits) const
{
  fml::print::printf("%.*f ", ndigits, (float)val);
}
#endif

template <typename REAL>
void unimat<REAL>::printval(const REAL val, uint8_t ndigits) const
{
  fml::print::printf("%.*f ", ndigits, val);
}


#endif
