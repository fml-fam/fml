// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_GPUVEC_H
#define FML_GPU_GPUVEC_H
#pragma once


#include <cstdint>
#include <cstdio>

#include "../_internals/univec.hh"

#include "arch/arch.hh"

#include "internals/gpuscalar.hh"
#include "internals/kernelfuns.hh"
#include "internals/launcher.hh"

#include "card.hh"


/**
  @brief Vector class for data held on a single GPU.
  
  @tparam T should be 'int', '__half', 'float' or 'double'.
 */
template <typename T>
class gpuvec : public univec<T>
{
  public:
    gpuvec(std::shared_ptr<card> gpu);
    gpuvec(std::shared_ptr<card> gpu, len_t size);
    gpuvec(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct=false);
    gpuvec(const gpuvec &x);
    ~gpuvec();
    
    void resize(len_t size);
    void resize(std::shared_ptr<card> gpu, len_t size);
    void inherit(std::shared_ptr<card> gpu);
    void inherit(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct=false);
    gpuvec<T> dupe() const;
    
    void print(uint8_t ndigits=4, bool add_final_blank=true) const;
    void info() const;
    
    void fill_zero();
    void fill_val(const T v);
    void fill_linspace(const T start, const T stop);
    
    void scale(const T s);
    void rev();
    
    T get(const len_t i) const;
    void set(const len_t i, const T v);
    
    bool operator==(const gpuvec<T> &x) const;
    bool operator!=(const gpuvec<T> &x) const;
    gpuvec<T>& operator=(const gpuvec<T> &x);
    
    std::shared_ptr<card> get_card() const {return c;};
    dim3 get_blockdim() const {return dim_block;};
    dim3 get_griddim() const {return dim_grid;};
    
  protected:
    std::shared_ptr<card> c;
  
  private:
    dim3 dim_block;
    dim3 dim_grid;
    
    void free();
    void check_params(len_t size);
    void check_gpu(std::shared_ptr<card> gpu);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
  @brief Construct vector object with no internal allocated storage.
  
  @param[in] gpu Shared pointer to GPU card object.
  
  @code
  auto c = gpuhelpers::new_card(0);
  gpuvec<float> x(c);
  @endcode
 */
template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu)
{
  check_gpu(gpu);
  
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



/**
  @brief Construct vector object with no internal allocated storage.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] size Number elements of the vector.
  
  @except If the allocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
  
  @code
  auto c = gpuhelpers::new_card(0);
  gpuvec<float> x(c, 5);
  @endcode
 */
template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu, len_t size)
{
  check_params(size);
  check_gpu(gpu);
  
  this->c = gpu;
  
  size_t len = (size_t) size * sizeof(T);
  this->data = (T*) this->c->mem_alloc(len);
  
  this->_size = size;
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
  
  this->free_data = true;
}



/**
  @brief Construct vector object with inherited data. Essentially the same as
  using the minimal constructor and immediately calling the `inherit()` method.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] data_ Storage array.
  @param[in] size Number elements of the array.
  @param[in] free_on_destruct Should the inherited array `data_` be freed when
  the vector object is destroyed?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu, T *data_, len_t size, bool free_on_destruct)
{
  check_params(size);
  check_gpu(gpu);
  
  this->c = gpu;
  
  this->_size = size;
  this->data = data_;
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
gpuvec<REAL>::gpuvec(const gpuvec<REAL> &x)
{
  this->_size = x.size();
  this->data = x.data_ptr();
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
  
  this->c = x.get_card();
  
  this->free_data = false;
}



template <typename T>
gpuvec<T>::~gpuvec()
{
  this->free();
  c = NULL;
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
void gpuvec<T>::resize(len_t size)
{
  check_params(size);
  
  if (this->_size == size)
    return;
  
  size_t len = (size_t) size * sizeof(T);
  
  T *realloc_ptr;
  realloc_ptr = (T*) this->c->mem_alloc(len);
  
  size_t oldlen = (size_t) this->_size * sizeof(T);
  size_t copylen = std::min(len, oldlen);
  this->c->mem_gpu2gpu(realloc_ptr, this->data, copylen);
  this->c->mem_free(this->data);
  this->data = realloc_ptr;
  
  this->_size = size;
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
}



/**
  @brief Resize the internal object storage.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] size Length of the vector needed.
  
  @allocs Resizing triggers a re-allocation.
  
  @except If the reallocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
 */
template <typename T>
void gpuvec<T>::resize(std::shared_ptr<card> gpu, len_t size)
{
  check_gpu(gpu);
  
  this->free();
  
  this->c = gpu;
  this->resize(size);
}



/**
  @brief Update the internal GPU card object to point to the specified one.
  
  @param[in] gpu Shared pointer to GPU card object.
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename T>
void gpuvec<T>::inherit(std::shared_ptr<card> gpu)
{
  check_gpu(gpu);
  
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



/**
  @brief Set the internal object storage to the specified array.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] data Value storage.
  @param[in] size Length of the vector. Should match the length of the input
  `data`.
  @param[in] free_on_destruct Should the object destructor free the internal
  array `data`?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename T>
void gpuvec<T>::inherit(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct)
{
  check_params(size);
  check_gpu(gpu);
  
  this->free();
  
  this->c = gpu;
  
  this->_size = size;
  this->data = data;
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
  
  this->free_data = free_on_destruct;
}



/// @brief Duplicate the object in a deep copy.
template <typename T>
gpuvec<T> gpuvec<T>::dupe() const
{
  gpuvec<T> cpy(this->c, this->_size);
  
  size_t len = (size_t) this->_size * sizeof(T);
  this->c->mem_gpu2gpu(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

/**
  @brief Copy data from a CPU object to another.
  
  @param[in] ndigits Number of decimal digits to print.
  @param[in] add_final_blank Should a final blank line be printed?
 */
template <typename REAL>
void gpuvec<REAL>::print(uint8_t ndigits, bool add_final_blank) const
{
  for (int i=0; i<this->_size; i++)
  {
    REAL tmp;
    this->c->mem_gpu2cpu(&tmp, this->data + i, sizeof(REAL));
    this->printval(tmp, ndigits);
  }
  
  if (add_final_blank)
    putchar('\n');
}



/// @brief Print some brief information about the object.
template <typename T>
void gpuvec<T>::info() const
{
  printf("# gpuvec ");
  printf("%d ", this->_size);
  printf("type=%s ", typeid(T).name());
  printf("\n");
}



// fillers

/// @brief Set all values to zero.
template <typename T>
void gpuvec<T>::fill_zero()
{
  size_t len = (size_t) this->_size * sizeof(T);
  this->c->mem_set(this->data, 0, len);
}



/**
  @brief Set all values to input value.
  
  @param[in] v Value to set all data values to.
 */
template <typename T>
void gpuvec<T>::fill_val(const T v)
{
  kernelfuns::kernel_fill_val<<<dim_grid, dim_block>>>(v, this->_size, 1, this->data);
  this->c->check();
}



/**
  @brief Set values to linearly spaced numbers.
  
  @param[in] start,stop Beginning/ending numbers.
 */
template <typename T>
void gpuvec<T>::fill_linspace(const T start, const T stop)
{
  kernelfuns::kernel_fill_linspace<<<dim_grid, dim_block>>>(start, stop, this->_size, 1, this->data);
  this->c->check();
}



/**
  @brief Multiply all values by the input value.
  
  @param[in] s Scaling value.
 */
template <typename T>
void gpuvec<T>::scale(const T s)
{
  kernelfuns::kernel_scale<<<dim_grid, dim_block>>>(s, this->_size, 1, this->data);
  this->c->check();
}



/// @brief Reverse the vector.
template <typename T>
void gpuvec<T>::rev()
{
  kernelfuns::kernel_rev_rows<<<dim_grid, dim_block>>>(this->_size, 1, this->data);
  this->c->check();
}



// operators

/**
  @brief Get the specified value.
  
  @param[in] i The index of the desired value, 0-indexed.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename T>
T gpuvec<T>::get(const len_t i) const
{
  this->check_index(i);

  T ret;
  this->c->mem_gpu2cpu(&ret, this->data + i, sizeof(T));
  return ret;
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i The index of the desired value, 0-indexed.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename T>
void gpuvec<T>::set(const len_t i, const T v)
{
  this->check_index(i);
  this->c->mem_cpu2gpu(this->data + i, &v, sizeof(T));
}



/**
  @brief See if the two objects are the same.
  
  @param[in] Comparison object.
  @return If the sizes mismatch, then `false` is necessarily returned. Next,
  if the pointer to the internal storage arrays match, then `true` is
  necessarily returned. Otherwise the objects are compared value by value.
 */
template <typename T>
bool gpuvec<T>::operator==(const gpuvec<T> &x) const
{
  if (this->_size != x.size())
    return false;
  else if (this->c->get_id() != x.get_card()->get_id())
    return false;
  else if (this->data == x.data_ptr())
    return true;
  
  int all_eq = 1;
  gpuscalar<int> all_eq_gpu(c, all_eq);
  
  kernelfuns::kernel_all_eq<<<dim_grid, dim_block>>>(this->_size, 1, this->data, x.data_ptr(), all_eq_gpu.data_ptr());
  
  all_eq_gpu.get_val(&all_eq);
  this->c->check();
  
  return (bool) all_eq;
}

/**
  @brief See if the two objects are not the same. Uses same internal logic as
  the `==` method.
  
  @param[in] Comparison object.
 */
template <typename T>
bool gpuvec<T>::operator!=(const gpuvec<T> &x) const
{
  return !(*this == x);
}



/**
  @brief Operator that sets the LHS to a shallow copy of the input. Desctruction
  of the LHS object will not result in the internal array storage being freed.
  
  @param[in] x Setter value.
 */
template <typename T>
gpuvec<T>& gpuvec<T>::operator=(const gpuvec<T> &x)
{
  this->c = x.get_card();
  this->_size = x.size();
  this->data = x.data_ptr();
  
  this->free_data = false;
  return *this;
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename T>
void gpuvec<T>::free()
{
  if (this->free_data && this->data)
  {
    this->c->mem_free(this->data);
    this->data = NULL;
  }
}



template <typename T>
void gpuvec<T>::check_params(len_t size)
{
  if (size < 0)
    throw std::runtime_error("invalid dimensions");
}



template <typename T>
void gpuvec<T>::check_gpu(std::shared_ptr<card> gpu)
{
  if (!gpu->valid_card())
    throw std::runtime_error("GPU card object is invalid");
}


#endif
