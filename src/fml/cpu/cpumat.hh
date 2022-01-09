// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_CPUMAT_H
#define FML_CPU_CPUMAT_H
#pragma once


#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "../_internals/arraytools/src/arraytools.hpp"

#include "../_internals/rand.hh"
#include "../_internals/omp.hh"
#include "../_internals/print.hh"
#include "../_internals/types.hh"
#include "../_internals/unimat.hh"

#include "cpuvec.hh"


namespace fml
{
  /**
    @brief Matrix class for data held on a single CPU.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  class cpumat : public fml::unimat<REAL>
  {
    public:
      cpumat();
      cpumat(len_t nrows, len_t ncols);
      cpumat(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
      cpumat(cpumat &&x);
      cpumat(const cpumat &x);
      ~cpumat();
      
      void resize(len_t nrows, len_t ncols);
      void inherit(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
      void inherit(cpumat<REAL> &data);
      cpumat<REAL> dupe() const;
      
      void print(uint8_t ndigits=4, bool add_final_blank=true) const;
      void info() const;
      
      void fill_zero();
      void fill_val(const REAL v);
      void fill_linspace();
      void fill_linspace(const REAL start, const REAL stop);
      void fill_eye();
      void fill_diag(const cpuvec<REAL> &v);
      void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
      void fill_runif(const REAL min=0, const REAL max=1);
      void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
      void fill_rnorm(const REAL mean=0, const REAL sd=1);
      
      void diag(cpuvec<REAL> &v);
      void antidiag(cpuvec<REAL> &v);
      void scale(const REAL s);
      void rev_rows();
      void rev_cols();
      
      bool any_inf() const;
      bool any_nan() const;
      
      REAL get(const len_t i) const;
      REAL get(const len_t i, const len_t j) const;
      void set(const len_t i, const REAL v);
      void set(const len_t i, const len_t j, const REAL v);
      void get_row(const len_t i, cpuvec<REAL> &v) const;
      void set_row(const len_t i, const cpuvec<REAL> &v);
      void get_col(const len_t j, cpuvec<REAL> &v) const;
      void set_col(const len_t i, const cpuvec<REAL> &v);
      
      bool operator==(const cpumat<REAL> &x) const;
      bool operator!=(const cpumat<REAL> &x) const;
      cpumat<REAL>& operator=(cpumat<REAL> &x);
      cpumat<REAL>& operator=(const cpumat<REAL> &x);
    
    protected:
      bool free_on_destruct() const {return this->free_data;};
      void dont_free_on_destruct() {this->free_data=false;};
    
    private:
      void free();
      void check_params(len_t nrows, len_t ncols);
  };
}



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
  @brief Construct matrix object with no internal allocated storage.
  
  @code
  cpumat<float> x();
  @endcode
 */
template <typename REAL>
fml::cpumat<REAL>::cpumat()
{
  this->m = 0;
  this->n = 0;
  this->data = NULL;
  
  this->free_data = true;
}



/**
  @brief Construct matrix object.
  
  @param[in] nrows,ncols Number rows/columns of the matrix.
  
  @except If the allocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
  
  @code
  cpumat<float> x(3, 2);
  @endcode
 */
template <typename REAL>
fml::cpumat<REAL>::cpumat(len_t nrows, len_t ncols)
{
  check_params(nrows, ncols);
  
  this->m = nrows;
  this->n = ncols;
  this->free_data = true;
  
  if (this->m == 0 || this->n == 0)
    return;
  
  size_t len = (size_t) nrows * ncols * sizeof(REAL);
  this->data = (REAL*) std::malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
}



/**
  @brief Construct matrix object with inherited data. Essentially the same as
  using the minimal constructor and immediately calling the `inherit()` method.
  
  @param[in] data_ Storage array.
  @param[in] nrows,ncols Number rows/columns of the array, i.e. the length of
  the array is nrows*ncols.
  @param[in] free_on_destruct Should the inherited array `data_` be freed when
  the matrix object is destroyed?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename REAL>
fml::cpumat<REAL>::cpumat(REAL *data_, len_t nrows, len_t ncols, bool free_on_destruct)
{
  check_params(nrows, ncols);
  
  this->m = nrows;
  this->n = ncols;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
fml::cpumat<REAL>::cpumat(cpumat<REAL> &&x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->free_data = x.free_on_destruct();
  x.dont_free_on_destruct();
}



template <typename REAL>
fml::cpumat<REAL>::cpumat(const cpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data.resize(this->m, this->n);
  
  size_t len = (size_t) this->m * this->n * sizeof(REAL);
  std::memcpy(this->data, x.data_ptr(), len);
  
  this->free_data = true;
}



template <typename REAL>
fml::cpumat<REAL>::~cpumat()
{
  this->free();
}



// memory management

/**
  @brief Resize the internal object storage.
  
  @param[in] nrows,ncols Number rows/columns needed.
  
  @allocs Resizing triggers a re-allocation.
  
  @except If the reallocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::resize(len_t nrows, len_t ncols)
{
  check_params(nrows, ncols);
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  const size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
  if ((nrows == 0 || ncols == 0) || (len == oldlen))
  {
    this->m = nrows;
    this->n = ncols;
    return;
  }
  
  void *realloc_ptr;
  if (oldlen == 0)
    realloc_ptr = malloc(len);
  else
    realloc_ptr = realloc(this->data, len);
  
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (REAL*) realloc_ptr;
  
  this->m = nrows;
  this->n = ncols;
}



/**
  @brief Set the internal object storage to the specified array.
  
  @param[in] data Value storage.
  @param[in] nrows,ncols Number rows/columns of the matrix. The product of
  these should be the length of the input `data`.
  @param[in] free_on_destruct Should the object destructor free the internal
  array `data`?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::inherit(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct)
{
  check_params(nrows, ncols);
  
  this->free();
  
  this->m = nrows;
  this->n = ncols;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



/// \overload
template <typename REAL>
void fml::cpumat<REAL>::inherit(cpumat<REAL> &data_)
{
  this->free();
  
  this->m = data_.nrows();
  this->n = data_.ncols();
  this->data = data_.data_ptr();
  
  this->free_data = false;
}



/// @brief Duplicate the object in a deep copy.
template <typename REAL>
fml::cpumat<REAL> fml::cpumat<REAL>::dupe() const
{
  fml::cpumat<REAL> cpy(this->m, this->n);
  
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  memcpy(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

/**
  @brief Print all values in the object.
  
  @param[in] ndigits Number of decimal digits to print.
  @param[in] add_final_blank Should a final blank line be printed?
 */
template <typename REAL>
void fml::cpumat<REAL>::print(uint8_t ndigits, bool add_final_blank) const
{
  for (len_t i=0; i<this->m; i++)
  {
    for (len_t j=0; j<this->n; j++)
      this->printval(this->data[i + this->m*j], ndigits);
    
    fml::print::putchar('\n');
  }
  
  if (add_final_blank)
    fml::print::putchar('\n');
}



/// @brief Print some brief information about the object.
template <typename REAL>
void fml::cpumat<REAL>::info() const
{
  fml::print::printf("# cpumat");
  fml::print::printf(" %dx%d", this->m, this->n);
  fml::print::printf(" type=%s", typeid(REAL).name());
  fml::print::printf("\n");
}



// fillers

/// @brief Set all values to zero.
template <typename REAL>
void fml::cpumat<REAL>::fill_zero()
{
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  memset(this->data, 0, len);
}



/**
  @brief Set all values to input value.
  
  @param[in] v Value to set all data values to.
 */
template <typename REAL>
void fml::cpumat<REAL>::fill_val(const REAL v)
{
  #pragma omp parallel for if((this->m)*(this->n) > fml::omp::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
  {
    #pragma omp simd
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] = v;
  }
}



/**
  @brief Set values to linearly spaced numbers.
  
  @param[in] start,stop Beginning/ending numbers. If not supplied, the matrix
  will be filled with whole numbers from 1 to the total number of elements.
 */
template <typename T>
void fml::cpumat<T>::fill_linspace()
{
  T start = 1;
  T stop = (T) (this->m * this->n);
  this->fill_linspace(start, stop);
}

template <typename REAL>
void fml::cpumat<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const REAL v = (stop-start)/((REAL) this->m*this->n - 1);
    
    #pragma omp parallel for if((this->m)*(this->n) > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<this->n; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<this->m; i++)
      {
        const len_t ind = i + this->m*j;
        this->data[ind] = v*((REAL) ind) + start;
      }
    }
  }
}

template <>
inline void fml::cpumat<int>::fill_linspace(const int start, const int stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const float v = (stop-start)/((float) this->m*this->n - 1);
    
    #pragma omp parallel for if((this->m)*(this->n) > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<this->n; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<this->m; i++)
      {
        const len_t ind = i + this->m*j;
        this->data[ind] = (int) roundf(v*((float) ind) + start);
      }
    }
  }
}



/// @brief Set diagonal entries to 1 and non-diagonal entries to 0.
template <typename REAL>
void fml::cpumat<REAL>::fill_eye()
{
  cpuvec<REAL> v(1);
  v.set(0, (REAL)1);
  this->fill_diag(v);
}



/**
  @brief Set diagonal entries of the matrix to those in the vector.
  
  @details If the vector is smaller than the matrix diagonal, the vector will
  recycle until the matrix diagonal is filled. If the vector is longer, then
  not all of it will be used.
  
  @param[in] v Vector of values to set the matrix diagonal to.
 */
template <typename REAL>
void fml::cpumat<REAL>::fill_diag(const cpuvec<REAL> &v)
{
  this->fill_zero();
  
  REAL *v_d = v.data_ptr();
  len_t min = std::min(this->m, this->n);
  
  #pragma omp for simd
  for (len_t i=0; i<min; i++)
    this->data[i + this->m*i] = v_d[i % v.size()];
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @param[in] seed Seed for the rng.
  @param[in] min,max Parameters for the generator.
 */
template <typename REAL>
void fml::cpumat<REAL>::fill_runif(const uint32_t seed, const REAL min, const REAL max)
{
  std::mt19937 mt(seed);
  static std::uniform_real_distribution<REAL> dist(min, max);
  
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] = dist(mt);
  }
}

/// \overload
template <typename REAL>
void fml::cpumat<REAL>::fill_runif(const REAL min, const REAL max)
{
  uint32_t seed = fml::rand::get_seed();
  this->fill_runif(seed, min, max);
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @param[in] seed Seed for the rng.
  @param[in] mean,sd Parameters for the generator.
 */
template <typename REAL>
void fml::cpumat<REAL>::fill_rnorm(const uint32_t seed, const REAL mean, const REAL sd)
{
  std::mt19937 mt(seed);
  static std::normal_distribution<REAL> dist(mean, sd);
  
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] = dist(mt);
  }
}

/// \overload
template <typename REAL>
void fml::cpumat<REAL>::fill_rnorm(const REAL mean, const REAL sd)
{
  uint32_t seed = fml::rand::get_seed();
  this->fill_rnorm(seed, mean, sd);
}



/**
  @brief Get the diagonal entries.
  
  @param[out] v The diagonal. Length should match the length of the diagonal
  of the input (minimum of the matrix dimensions). If not, the vector will
  automatically be resized.
  
  @allocs If the output dimension is inappropriately sized, it will
  automatically be re-allocated.
  
  @except If a reallocation is triggered and fails, a `bad_alloc` exception
  will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::diag(cpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp parallel for simd if(minmn > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<minmn; i++)
    v_ptr[i] = this->data[i + this->m*i];
}



/**
  @brief Get the anti-diagonal entries, i.e. those on the bottom-left to
  top-right.
  
  @param[out] v The anti-diagonal. Length should match the length of the
  diagonal of the input (minimum of the matrix dimensions). If not, the vector
  will automatically be resized.
  
  @allocs If the output dimension is inappropriately sized, it will
  automatically be re-allocated.
  
  @except If a reallocation is triggered and fails, a `bad_alloc` exception
  will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::antidiag(cpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp parallel for simd if(minmn > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<minmn; i++)
    v_ptr[i] = this->data[(this->m-1-i) + this->m*i];
}



/**
  @brief Multiply all values by the input value.
  
  @param[in] s Scaling value.
 */
template <typename REAL>
void fml::cpumat<REAL>::scale(const REAL s)
{
  #pragma omp parallel for if((this->m)*(this->n) > fml::omp::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
  {
    #pragma omp simd
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] *= s;
  }
}



/// @brief Reverse the rows of the matrix.
template <typename REAL>
void fml::cpumat<REAL>::rev_rows()
{
  for (len_t j=0; j<this->n; j++)
  {
    len_t last = this->m - 1;
    for (len_t i=0; i<this->m/2; i++)
    {
      const REAL tmp = this->data[i + this->m*j];
      this->data[i + this->m*j] = this->data[last + this->m*j];
      this->data[last + this->m*j] = tmp;
    }
    
    last--;
  }
}



/// @brief Reverse the columns of the matrix.
template <typename REAL>
void fml::cpumat<REAL>::rev_cols()
{
  len_t last = this->n - 1;
  
  for (len_t j=0; j<this->n/2; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      const REAL tmp = this->data[i + this->m*j];
      this->data[i + this->m*j] = this->data[i + this->m*last];
      this->data[i + this->m*last] = tmp;
    }
    
    last--;
  }
}



/// @brief Are any values infinite?
template <typename REAL>
bool fml::cpumat<REAL>::any_inf() const
{
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      if (isinf(this->data[i + this->m*j]))
        return true;
    }
  }
  
  return false;
}



/// @brief Are any values NaN?
template <typename REAL>
bool fml::cpumat<REAL>::any_nan() const
{
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      if (isnan(this->data[i + this->m*j]))
        return true;
    }
  }
  
  return false;
}



// operators

/**
  @brief Get the specified value.
  
  @param[in] i The index of the desired value, 0-indexed. The numbering
  considers the internal storage as a 1-dimensional array.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename REAL>
REAL fml::cpumat<REAL>::get(const len_t i) const
{
  this->check_index(i);
  return this->data[i];
}

/**
  @brief Get the specified value.
  
  @param[in] i,j The indices of the desired value, 0-indexed.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename REAL>
REAL fml::cpumat<REAL>::get(const len_t i, const len_t j) const
{
  this->check_index(i, j);
  return this->data[i + (this->m)*j];
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i The index of the desired value, 0-indexed. The numbering
  considers the internal storage as a 1-dimensional array.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename REAL>
void fml::cpumat<REAL>::set(const len_t i, const REAL v)
{
  this->check_index(i);
  this->data[i] = v;
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i,j The indices of the desired value, 0-indexed.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename REAL>
void fml::cpumat<REAL>::set(const len_t i, const len_t j, const REAL v)
{
  this->check_index(i, j);
  this->data[i + (this->m)*j] = v;
}



/**
  @brief Get the specified row.
  
  @param[in] i The desired row, 0-indexed.
  @param[out] v The row values.
  
  @allocs If the output dimension is inappropriately sized, it will
  automatically be re-allocated.
  
  @except If `i` is an inappropriate value (i.e. does not refer to a matrix
  row), then the method will throw a `logic_error` exception. If a reallocation
  is triggered and fails, a `bad_alloc` exception will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::get_row(const len_t i, cpuvec<REAL> &v) const
{
  if (i < 0 || i >= this->m)
    throw std::logic_error("invalid matrix row");
  
  v.resize(this->n);
  REAL *v_d = v.data_ptr();
  
  #pragma omp parallel for simd if(this->n > fml::omp::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
    v_d[j] = this->data[i + this->m*j];
}



/**
  @brief Set the specified row.
  
  @param[in] i The desired row, 0-indexed.
  @param[in] v The row values.
  
  @except If `i` is an inappropriate value (i.e. does not refer to a matrix
  row), then the method will throw a `logic_error` exception. If the vector
  is inappropriately sized, a `runtime_error` exception will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::set_row(const len_t i, const cpuvec<REAL> &v)
{
  if (i < 0 || i >= this->m)
    throw std::logic_error("invalid matrix row");
  if (v.size() != this->n)
    throw std::runtime_error("non-conformable arguments");
  
  REAL *v_d = v.data_ptr();
  #pragma omp parallel for simd if(this->n > fml::omp::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
    this->data[i + this->m*j] = v_d[j];
}



/**
  @brief Get the specified column.
  
  @param[in] j The desired column, 0-indexed.
  @param[out] v The column values.
  
  @allocs If the output dimension is inappropriately sized, it will
  automatically be re-allocated.
  
  @except If `j` is an inappropriate value (i.e. does not refer to a matrix
  column), then the method will throw a `logic_error` exception. If a
  reallocation is triggered and fails, a `bad_alloc` exception will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::get_col(const len_t j, cpuvec<REAL> &v) const
{
  if (j < 0 || j >= this->n)
    throw std::logic_error("invalid matrix column");
  
  v.resize(this->m);
  REAL *v_d = v.data_ptr();
  
  #pragma omp parallel for if(this->m > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<this->m; i++)
    v_d[i] = this->data[i + this->m*j];
}



/**
  @brief Set the specified column.
  
  @param[in] j The desired column, 0-indexed.
  @param[in] v The column values.
  
  @except If `i` is an inappropriate value (i.e. does not refer to a matrix
  row), then the method will throw a `logic_error` exception. If the vector
  is inappropriately sized, a `runtime_error` exception will be thrown.
 */
template <typename REAL>
void fml::cpumat<REAL>::set_col(const len_t j, const cpuvec<REAL> &v)
{
  if (j < 0 || j >= this->n)
    throw std::logic_error("invalid matrix column");
  if (v.size() != this->m)
    throw std::runtime_error("non-conformable arguments");
  
  REAL *v_d = v.data_ptr();
  #pragma omp parallel for simd if(this->n > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<this->m; i++)
    this->data[i + this->m*j] = v_d[i];
}



/**
  @brief See if the two objects are the same.
  
  @param[in] Comparison object.
  @return If the dimensions mismatch, then `false` is necessarily returned.
  Next, if the pointer to the internal storage arrays match, then `true` is
  necessarily returned. Otherwise the objects are compared value by value.
 */
template <typename REAL>
bool fml::cpumat<REAL>::operator==(const fml::cpumat<REAL> &x) const
{
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  else if (this->data == x.data_ptr())
    return true;
  
  const REAL *x_d = x.data_ptr();
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      const REAL a = this->data[i + this->m*j];
      const REAL b = x_d[i + this->m*j];
      if (!arraytools::fltcmp::eq(a, b))
        return false;
    }
  }
  
  return true;
}

/**
  @brief See if the two objects are not the same. Uses same internal logic as
  the `==` method.
  
  @param[in] Comparison object.
 */
template <typename REAL>
bool fml::cpumat<REAL>::operator!=(const fml::cpumat<REAL> &x) const
{
  return !(*this == x);
}



/**
  @brief Operator that sets the LHS to a shallow copy of the input. Desctruction
  of the LHS object will not result in the internal array storage being freed.
  
  @param[in] x Setter value.
 */
template <typename REAL>
fml::cpumat<REAL>& fml::cpumat<REAL>::operator=(fml::cpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->free_data = x.free_on_destruct();
  x.dont_free_on_destruct();
  
  return *this;
}

/// \overload
template <typename REAL>
fml::cpumat<REAL>& fml::cpumat<REAL>::operator=(const fml::cpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->free_data = false;
  
  return *this;
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename REAL>
void fml::cpumat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void fml::cpumat<REAL>::check_params(len_t nrows, len_t ncols)
{
  if (nrows < 0 || ncols < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
