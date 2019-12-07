// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_GPUVEC_H
#define FML_GPU_GPUVEC_H
#pragma once


#include <cstdint>
#include <cstdio>

#include "../_internals/univec.hh"

#include "internals/launcher.hh"

#include "card.hh"
#include "kernelfuns.hh"


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
    void fill_one();
    void fill_val(const T v);
    void fill_linspace(const T start, const T stop);
    
    void scale(const T s);
    void rev();
    
    T get(len_t i) const;
    void set(len_t i, T v);
    
    bool operator==(const gpuvec<T> &x) const;
    bool operator!=(const gpuvec<T> &x) const;
    
    gpuvec<T>& operator=(const gpuvec<T> &x);
    
    std::shared_ptr<card> get_card() const {return c;};
    dim3 get_blockdim() const {return dim_block;};
    dim3 get_griddim() const {return dim_grid;};
    
  protected:
    std::shared_ptr<card> c;
  
  private:
    void free();
    void check_params(len_t size);
    dim3 dim_block;
    dim3 dim_grid;
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu)
{
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu, len_t size)
{
  check_params(size);
  
  this->c = gpu;
  
  size_t len = (size_t) size * sizeof(T);
  this->data = (T*) this->c->mem_alloc(len);
  
  this->_size = size;
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
  
  this->free_data = true;
}



template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu, T *data_, len_t size, bool free_on_destruct)
{
  check_params(size);
  
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
}



// memory management

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



template <typename T>
void gpuvec<T>::resize(std::shared_ptr<card> gpu, len_t size)
{
  this->c = gpu;
  this->resize(size);
}



template <typename T>
void gpuvec<T>::inherit(std::shared_ptr<card> gpu)
{
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename T>
void gpuvec<T>::inherit(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct)
{
  check_params(size);
  
  this->free();
  
  this->c = gpu;
  
  this->_size = size;
  this->data = data;
  
  dim_block = kernel_launcher::dim_block1();
  dim_grid = kernel_launcher::dim_grid(this->_size);
  
  this->free_data = free_on_destruct;
}



template <typename T>
gpuvec<T> gpuvec<T>::dupe() const
{
  gpuvec<T> cpy(this->c, this->_size);
  
  size_t len = (size_t) this->_size * sizeof(T);
  this->c->mem_gpu2gpu(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

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



template <typename T>
void gpuvec<T>::info() const
{
  printf("# gpuvec ");
  printf("%d ", this->_size);
  printf("type=%s ", typeid(T).name());
  printf("\n");
}



// fillers

template <typename T>
void gpuvec<T>::fill_zero()
{
  size_t len = (size_t) this->_size * sizeof(T);
  this->c->mem_set(this->data, 0, len);
}



template <typename T>
void gpuvec<T>::fill_one()
{
  this->fill_val((T) 1);
}



template <typename T>
void gpuvec<T>::fill_val(const T v)
{
  kernelfuns::kernel_fill_val<<<dim_grid, dim_block>>>(v, this->_size, 1, this->data);
  this->c->check();
}



template <typename T>
void gpuvec<T>::fill_linspace(const T start, const T stop)
{
  kernelfuns::kernel_fill_linspace<<<dim_grid, dim_block>>>(start, stop, this->_size, 1, this->data);
  this->c->check();
}



template <typename T>
void gpuvec<T>::scale(const T s)
{
  kernelfuns::kernel_scale<<<dim_grid, dim_block>>>(s, this->_size, 1, this->data);
  this->c->check();
}



template <typename T>
void gpuvec<T>::rev()
{
  kernelfuns::kernel_rev_rows<<<dim_grid, dim_block>>>(this->_size, 1, this->data);
  this->c->check();
}



// operators

template <typename T>
T gpuvec<T>::get(len_t i) const
{
  this->check_index(i);

  T ret;
  this->c->mem_gpu2cpu(&ret, this->data + i, sizeof(T));
  return ret;
}

template <typename T>
void gpuvec<T>::set(len_t i, T v)
{
  this->check_index(i);
  this->c->mem_cpu2gpu(this->data + i, &v, sizeof(T));
}



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
  int *all_eq_gpu = (int*) this->c->mem_alloc(sizeof(*all_eq_gpu));
  this->c->mem_cpu2gpu(all_eq_gpu, &all_eq, sizeof(all_eq));
  
  kernelfuns::kernel_all_eq<<<dim_grid, dim_block>>>(this->_size, 1, this->data, x.data_ptr(), all_eq_gpu);
  
  this->c->mem_gpu2cpu(&all_eq, all_eq_gpu, sizeof(all_eq));
  this->c->mem_free(all_eq_gpu);
  
  this->c->check();
  
  return (bool) all_eq;
}

template <typename T>
bool gpuvec<T>::operator!=(const gpuvec<T> &x) const
{
  return !(*this == x);
}



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

template <typename REAL>
void gpuvec<REAL>::free()
{
  if (this->free_data && this->data)
  {
    this->c->mem_free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void gpuvec<REAL>::check_params(len_t size)
{
  if (size < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
