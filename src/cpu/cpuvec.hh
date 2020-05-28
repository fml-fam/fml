// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_CPUVEC_H
#define FML_CPU_CPUVEC_H
#pragma once


#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "../_internals/arraytools/src/arraytools.hpp"

#include "../_internals/omp.hh"
#include "../_internals/print.hh"
#include "../_internals/univec.hh"


/**
  @brief Vector class for data held on a single CPU.
  
  @tparam T should be 'int', 'float' or 'double'.
 */
template <typename T>
class cpuvec : public univec<T>
{
  public:
    cpuvec();
    cpuvec(len_t size);
    cpuvec(T *data, len_t size, bool free_on_destruct=false);
    cpuvec(const cpuvec &x);
    ~cpuvec();
    
    void resize(len_t size);
    void inherit(T *data, len_t size, bool free_on_destruct=false);
    cpuvec<T> dupe() const;
    
    void print(uint8_t ndigits=4, bool add_final_blank=true) const;
    void info() const;
    
    void fill_zero();
    void fill_val(const T v);
    void fill_linspace(const T start, const T stop);
    
    void scale(const T s);
    void rev();
    T sum();
    
    T get(const len_t i) const;
    void set(const len_t i, const T v);
    
    bool operator==(const cpuvec<T> &x) const;
    bool operator!=(const cpuvec<T> &x) const;
    cpuvec<T>& operator=(const cpuvec<T> &x);
  
  private:
    void free();
    void check_params(len_t size);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
  @brief Construct vector object with no internal allocated storage.
  
  @code
  cpuvec<float> x();
  @endcode
 */
template <typename T>
cpuvec<T>::cpuvec()
{
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



/**
  @brief Construct vector object with no internal allocated storage.
  
  @param[in] size Number elements of the vector.
  
  @except If the allocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
  
  @code
  cpuvec<float> x(5);
  @endcode
 */
template <typename T>
cpuvec<T>::cpuvec(len_t size)
{
  check_params(size);
  
  this->_size = size;
  this->free_data = true;
  
  if (size == 0)
    return;
  
  const size_t len = (size_t) size * sizeof(T);
  this->data = (T*) std::malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
}



/**
  @brief Construct vector object with inherited data. Essentially the same as
  using the minimal constructor and immediately calling the `inherit()` method.
  
  @param[in] data_ Storage array.
  @param[in] size Number elements of the array.
  @param[in] free_on_destruct Should the inherited array `data_` be freed when
  the vector object is destroyed?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename T>
cpuvec<T>::cpuvec(T *data_, len_t size, bool free_on_destruct)
{
  check_params(size);
  
  this->_size = size;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename T>
cpuvec<T>::cpuvec(const cpuvec<T> &x)
{
  this->_size = x.size();
  this->data = x.data_ptr();
  
  this->free_data = false;
}



template <typename T>
cpuvec<T>::~cpuvec()
{
  this->free();
}



// memory management

/**
  @brief Resize the internal object storage.
  
  @param[in] size Length of the vector needed.
  
  @allocs Resizing triggers a re-allocation.
  
  @except If the reallocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
 */
template <typename T>
void cpuvec<T>::resize(len_t size)
{
  check_params(size);
  
  if (size == 0)
  {
    this->_size = size;
    return;
  }
  else if (this->_size == size)
    return;
  
  const size_t len = (size_t) size * sizeof(T);
  
  void *realloc_ptr;
  if (this->_size == 0)
    realloc_ptr = malloc(len);
  else
    realloc_ptr = realloc(this->data, len);
  
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (T*) realloc_ptr;
  this->_size = size;
}



/**
  @brief Set the internal object storage to the specified array.
  
  @param[in] data Value storage.
  @param[in] size Length of the vector. Should match the length of the input
  `data`.
  @param[in] free_on_destruct Should the object destructor free the internal
  array `data`?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename T>
void cpuvec<T>::inherit(T *data, len_t size, bool free_on_destruct)
{
  check_params(size);
  
  this->free();
  
  this->_size = size;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



/// @brief Duplicate the object in a deep copy.
template <typename T>
cpuvec<T> cpuvec<T>::dupe() const
{
  cpuvec<T> cpy(this->_size);
  
  const size_t len = (size_t) this->_size * sizeof(T);
  memcpy(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

/**
  @brief Copy data from a CPU object to another.
  
  @param[in] ndigits Number of decimal digits to print.
  @param[in] add_final_blank Should a final blank line be printed?
 */
template <typename T>
void cpuvec<T>::print(uint8_t ndigits, bool add_final_blank) const
{
  for (len_t i=0; i<this->_size; i++)
    this->printval(this->data[i], ndigits);
  
  fml::print::putchar('\n');
  if (add_final_blank)
    fml::print::putchar('\n');
}



/// @brief Print some brief information about the object.
template <typename T>
void cpuvec<T>::info() const
{
  fml::print::printf("# cpuvec");
  fml::print::printf(" %d", this->_size);
  fml::print::printf(" type=%s", typeid(T).name());
  fml::print::printf("\n");
}



// fillers

/// @brief Set all values to zero.
template <typename T>
void cpuvec<T>::fill_zero()
{
  const size_t len = (size_t) this->_size * sizeof(T);
  memset(this->data, 0, len);
}



/**
  @brief Set all values to input value.
  
  @param[in] v Value to set all data values to.
 */
template <typename T>
void cpuvec<T>::fill_val(const T v)
{
  #pragma omp parallel for simd if(this->_size > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<this->_size; i++)
    this->data[i] = v;
}



/**
  @brief Set values to linearly spaced numbers.
  
  @param[in] start,stop Beginning/ending numbers.
 */
template <typename REAL>
void cpuvec<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const REAL v = (stop-start)/((REAL) this->_size - 1);
    
    #pragma omp parallel for simd if(this->_size > fml::omp::OMP_MIN_SIZE)
    for (len_t i=0; i<this->_size; i++)
      this->data[i] = v*((REAL) i) + start;
  }
}

template <>
inline void cpuvec<int>::fill_linspace(const int start, const int stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const float v = (stop-start)/((float) this->_size - 1);
    
    #pragma omp parallel for simd if(this->_size > fml::omp::OMP_MIN_SIZE)
    for (len_t i=0; i<this->_size; i++)
      this->data[i] = (int) roundf(v*((float) i) + start);
  }
}



/**
  @brief Multiply all values by the input value.
  
  @param[in] s Scaling value.
 */
template <typename T>
void cpuvec<T>::scale(const T s)
{
  #pragma omp parallel for simd if(this->_size > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<this->_size; i++)
    this->data[i] *= s;
}



/// @brief Reverse the vector.
template <typename T>
void cpuvec<T>::rev()
{
  len_t j = this->_size - 1;
  
  for (len_t i=0; i<this->_size/2; i++)
  {
    const T tmp = this->data[i];
    this->data[i] = this->data[j];
    this->data[j] = tmp;
    j--;
  }
}



/// @brief Sum the vector.
template <typename T>
T cpuvec<T>::sum()
{
  T s = 0;
  
  #pragma omp for simd
  for (len_t i=0; i<this->_size; i++)
    s += this->data[i];
  
  return s;
}



// operators

/**
  @brief Get the specified value.
  
  @param[in] i The index of the desired value, 0-indexed.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename T>
T cpuvec<T>::get(const len_t i) const
{
  this->check_index(i);
  return this->data[i];
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i The index of the desired value, 0-indexed.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename T>
void cpuvec<T>::set(const len_t i, const T v)
{
  this->check_index(i);
  this->data[i] = v;
}



/**
  @brief See if the two objects are the same.
  
  @param[in] Comparison object.
  @return If the sizes mismatch, then `false` is necessarily returned. Next,
  if the pointer to the internal storage arrays match, then `true` is
  necessarily returned. Otherwise the objects are compared value by value.
 */
template <typename T>
bool cpuvec<T>::operator==(const cpuvec<T> &x) const
{
  if (this->_size != x.size())
    return false;
  else if (this->data == x.data_ptr())
    return true;
  
  const T *x_d = x.data_ptr();
  for (len_t i=0; i<this->_size; i++)
  {
    const T a = this->data[i];
    const T b = x_d[i];
    if (!arraytools::fltcmp::eq(a, b))
      return false;
  }
  
  return true;
}

/**
  @brief See if the two objects are not the same. Uses same internal logic as
  the `==` method.
  
  @param[in] Comparison object.
 */
template <typename T>
bool cpuvec<T>::operator!=(const cpuvec<T> &x) const
{
  return !(*this == x);
}



/**
  @brief Operator that sets the LHS to a shallow copy of the input. Desctruction
  of the LHS object will not result in the internal array storage being freed.
  
  @param[in] x Setter value.
 */
template <typename T>
cpuvec<T>& cpuvec<T>::operator=(const cpuvec<T> &x)
{
  this->_size = x.size();
  this->data = x.data_ptr();
  
  this->free_data = false;
  return *this;
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename T>
void cpuvec<T>::free()
{
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void cpuvec<REAL>::check_params(len_t size)
{
  if (size < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
