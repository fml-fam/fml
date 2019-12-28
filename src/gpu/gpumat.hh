// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_GPUMAT_H
#define FML_GPU_GPUMAT_H
#pragma once


#include <cstdint>
#include <cstdio>

#include "../_internals/types.hh"
#include "../_internals/unimat.hh"

#include "arch/arch.hh"

#include "internals/gpuscalar.hh"
#include "internals/kernelfuns.hh"
#include "internals/launcher.hh"

#include "card.hh"
#include "gpuvec.hh"


/**
  @brief Matrix class for data held on a single GPU. 
  
  @tparam REAL should be '__half', 'float', or 'double'.
 */
template <typename REAL>
class gpumat : public unimat<REAL>
{
  public:
    gpumat(std::shared_ptr<card> gpu);
    gpumat(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    gpumat(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat(const gpumat &x);
    ~gpumat();
    
    void resize(len_t nrows, len_t ncols);
    void resize(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    void inherit(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat<REAL> dupe() const;
    
    void print(uint8_t ndigits=4, bool add_final_blank=true) const;
    void info() const;
    
    void fill_zero();
    void fill_val(const REAL v);
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_diag(const gpuvec<REAL> &v);
    void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
    void fill_runif(const REAL min=0, const REAL max=1);
    void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
    void fill_rnorm(const REAL mean=0, const REAL sd=1);
    
    void diag(gpuvec<REAL> &v);
    void antidiag(gpuvec<REAL> &v);
    void scale(const REAL s);
    void rev_rows();
    void rev_cols();
    
    bool any_inf() const;
    bool any_nan() const;
    
    REAL get(const len_t i) const;
    REAL get(const len_t i, const len_t j) const;
    void set(const len_t i, const REAL v);
    void set(const len_t i, const len_t j, const REAL v);
    void get_row(const len_t i, gpuvec<REAL> &v) const;
    void get_col(const len_t j, gpuvec<REAL> &v) const;
    
    bool operator==(const gpumat<REAL> &x) const;
    bool operator!=(const gpumat<REAL> &x) const;
    gpumat<REAL>& operator=(const gpumat<REAL> &x);
    
    std::shared_ptr<card> get_card() const {return c;};
    dim3 get_blockdim() const {return dim_block;};
    dim3 get_griddim() const {return dim_grid;};
    
  protected:
    std::shared_ptr<card> c;
  
  private:
    dim3 dim_block;
    dim3 dim_grid;
    
    void free();
    void check_params(len_t nrows, len_t ncols);
    void check_gpu(std::shared_ptr<card> gpu);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
  @brief Construct matrix object with no internal allocated storage.
  
  @param[in] gpu Shared pointer to GPU card object.
  
  @code
  auto c = gpuhelpers::new_card(0);
  gpumat<float> x(c);
  @endcode
 */
template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu)
{
  check_gpu(gpu);
  
  this->c = gpu;
  
  this->m = 0;
  this->n = 0;
  this->data = NULL;
  
  this->free_data = true;
}



/**
  @brief Construct matrix object.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] nrows,ncols Number rows/columns of the matrix.
  
  @except If the allocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
  
  @code
  auto c = gpuhelpers::new_card(0);
  gpumat<float> x(c, 3, 2);
  @endcode
 */
template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu, len_t nrows, len_t ncols)
{
  check_params(nrows, ncols);
  check_gpu(gpu);
  
  this->c = gpu;
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  this->data = (REAL*) this->c->mem_alloc(len);
  
  this->m = nrows;
  this->n = ncols;
  
  dim_block = fml::kernel_launcher::dim_block2();
  dim_grid = fml::kernel_launcher::dim_grid(this->m, this->n);
  
  this->free_data = true;
}



/**
  @brief Construct matrix object with inherited data. Essentially the same as
  using the minimal constructor and immediately calling the `inherit()` method.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] data_ Storage array.
  @param[in] nrows,ncols Number rows/columns of the array, i.e. the length of
  the array is nrows*ncols.
  @param[in] free_on_destruct Should the inherited array `data_` be freed when
  the matrix object is destroyed?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu, REAL *data_, len_t nrows, len_t ncols, bool free_on_destruct)
{
  check_params(nrows, ncols);
  check_gpu(gpu);
  
  this->c = gpu;
  
  this->m = nrows;
  this->n = ncols;
  this->data = data_;
  
  dim_block = fml::kernel_launcher::dim_block2();
  dim_grid = fml::kernel_launcher::dim_grid(this->m, this->n);
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
gpumat<REAL>::gpumat(const gpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  dim_block = fml::kernel_launcher::dim_block2();
  dim_grid = fml::kernel_launcher::dim_grid(this->m, this->n);
  
  this->c = x.get_card();
  
  this->free_data = false;
}



template <typename REAL>
gpumat<REAL>::~gpumat()
{
  this->free();
  c = NULL;
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
void gpumat<REAL>::resize(len_t nrows, len_t ncols)
{
  check_params(nrows, ncols);
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  const size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
  if (len == oldlen)
  {
    this->m = nrows;
    this->n = ncols;
    return;
  }
  
  REAL *realloc_ptr;
  realloc_ptr = (REAL*) this->c->mem_alloc(len);
  
  const size_t copylen = std::min(len, oldlen);
  this->c->mem_gpu2gpu(realloc_ptr, this->data, copylen);
  this->c->mem_free(this->data);
  this->data = realloc_ptr;
  
  this->m = nrows;
  this->n = ncols;
  
  dim_block = fml::kernel_launcher::dim_block2();
  dim_grid = fml::kernel_launcher::dim_grid(this->m, this->n);
}



/**
  @brief Resize the internal object storage.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] nrows,ncols Number rows/columns needed.
  
  @allocs Resizing triggers a re-allocation.
  
  @except If the reallocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
 */
template <typename REAL>
void gpumat<REAL>::resize(std::shared_ptr<card> gpu, len_t nrows, len_t ncols)
{
  check_gpu(gpu);
  
  this->c = gpu;
  this->resize(nrows, ncols);
}



/**
  @brief Set the internal object storage to the specified array.
  
  @param[in] gpu Shared pointer to GPU card object.
  @param[in] data Value storage.
  @param[in] nrows,ncols Number rows/columns of the matrix. The product of
  these should be the length of the input `data`.
  @param[in] free_on_destruct Should the object destructor free the internal
  array `data`?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
 */
template <typename REAL>
void gpumat<REAL>::inherit(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct)
{
  check_params(nrows, ncols);
  check_gpu(gpu);
  
  this->free();
  
  this->c = gpu;
  
  this->m = nrows;
  this->n = ncols;
  this->data = data;
  
  dim_block = fml::kernel_launcher::dim_block2();
  dim_grid = fml::kernel_launcher::dim_grid(this->m, this->n);
  
  this->free_data = free_on_destruct;
}



/// @brief Duplicate the object in a deep copy.
template <typename REAL>
gpumat<REAL> gpumat<REAL>::dupe() const
{
  gpumat<REAL> cpy(this->c, this->m, this->n);
  
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  this->c->mem_gpu2gpu(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

/**
  @brief Print all values in the object.
  
  @param[in] ndigits Number of decimal digits to print.
  @param[in] add_final_blank Should a final blank line be printed?
 */
template <typename REAL>
void gpumat<REAL>::print(uint8_t ndigits, bool add_final_blank) const
{
  for (int i=0; i<this->m; i++)
  {
    for (int j=0; j<this->n; j++)
    {
      REAL tmp;
      this->c->mem_gpu2cpu(&tmp, this->data + (i + this->m*j), sizeof(REAL));
      this->printval(tmp, ndigits);
    }
  
    putchar('\n');
  }
  
  if (add_final_blank)
    putchar('\n');
}



/// @brief Print some brief information about the object.
template <typename REAL>
void gpumat<REAL>::info() const
{
  printf("# gpumat ");
  printf("%dx%d ", this->m, this->n);
  printf("type=%s ", typeid(REAL).name());
  printf("\n");
}



// fillers

/// @brief Set all values to zero.
template <typename REAL>
void gpumat<REAL>::fill_zero()
{
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  this->c->mem_set(this->data, 0, len);
}



/**
  @brief Set all values to input value.
  
  @param[in] v Value to set all data values to.
 */
template <typename REAL>
void gpumat<REAL>::fill_val(const REAL v)
{
  fml::kernelfuns::kernel_fill_val<<<dim_grid, dim_block>>>(v, this->m, this->n, this->data);
  this->c->check();
}



/**
  @brief Set values to linearly spaced numbers.
  
  @param[in] start,stop Beginning/ending numbers.
 */
template <typename REAL>
void gpumat<REAL>::fill_linspace(REAL start, REAL stop)
{
  // if (start == stop)
  //   this->fill_val(start);
  // else
  {
    fml::kernelfuns::kernel_fill_linspace<<<dim_grid, dim_block>>>(start, stop, this->m, this->n, this->data);
    this->c->check();
  }
}



/// @brief Set diagonal entries to 1 and non-diagonal entries to 0.
template <typename REAL>
void gpumat<REAL>::fill_eye()
{
  fml::kernelfuns::kernel_fill_eye<<<dim_grid, dim_block>>>(this->m, this->n, this->data);
  this->c->check();
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @details If the vector is smaller than the matrix diagonal, the vector will
  recycle until the matrix diagonal is filled. If the vector is longer, then
  not all of it will be used.
  
  @param[in] v Vector of values to set the matrix diagonal to.
 */
template <typename REAL>
void gpumat<REAL>::fill_diag(const gpuvec<REAL> &v)
{
  fml::kernelfuns::kernel_fill_diag<<<dim_grid, dim_block>>>(v.size(), v.data_ptr(), this->m, this->n, this->data);
  this->c->check();
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @param[in] seed Seed for the rng.
  @param[in] min,max Parameters for the generator.
 */
template <typename REAL>
void gpumat<REAL>::fill_runif(uint32_t seed, REAL min, REAL max)
{
  
}

/// \overload
template <typename REAL>
void gpumat<REAL>::fill_runif(REAL min, REAL max)
{
  
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @param[in] seed Seed for the rng.
  @param[in] mean,sd Parameters for the generator.
 */
template <typename REAL>
void gpumat<REAL>::fill_rnorm(uint32_t seed, REAL mean, REAL sd)
{
  
}

/// \overload
template <typename REAL>
void gpumat<REAL>::fill_rnorm(REAL mean, REAL sd)
{
  
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
void gpumat<REAL>::diag(gpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  
  fml::kernelfuns::kernel_diag<<<dim_grid, dim_block>>>(this->m, this->n, this->data, v.data_ptr());
  this->c->check();
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
void gpumat<REAL>::antidiag(gpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  
  fml::kernelfuns::kernel_antidiag<<<dim_grid, dim_block>>>(this->m, this->n, this->data, v.data_ptr());
  this->c->check();
}



/**
  @brief Multiply all values by the input value.
  
  @param[in] s Scaling value.
 */
template <typename REAL>
void gpumat<REAL>::scale(const REAL s)
{
  fml::kernelfuns::kernel_scale<<<dim_grid, dim_block>>>(s, this->m, this->n, this->data);
  this->c->check();
}



/// @brief Reverse the rows of the matrix.
template <typename REAL>
void gpumat<REAL>::rev_rows()
{
  fml::kernelfuns::kernel_rev_rows<<<dim_grid, dim_block>>>(this->m, this->n, this->data);
  this->c->check();
}



/// @brief Reverse the columns of the matrix.
template <typename REAL>
void gpumat<REAL>::rev_cols()
{
  fml::kernelfuns::kernel_rev_cols<<<dim_grid, dim_block>>>(this->m, this->n, this->data);
  this->c->check();
}



/// @brief Are any values infinite?
template <typename REAL>
bool gpumat<REAL>::any_inf() const
{
  int has_inf = 0;
  gpuscalar<int> has_inf_gpu(c, has_inf);
  
  fml::kernelfuns::kernel_any_inf<<<dim_grid, dim_block>>>(this->m, this->n, this->data, has_inf_gpu.data_ptr());
  
  has_inf_gpu.get_val(&has_inf);
  this->c->check();
  
  return (bool) has_inf;
}



template <typename REAL>
bool gpumat<REAL>::any_nan() const
{
  int has_nan = 0;
  gpuscalar<int> has_nan_gpu(c, has_nan);
  
  fml::kernelfuns::kernel_any_nan<<<dim_grid, dim_block>>>(this->m, this->n, this->data, has_nan_gpu.data_ptr());
  
  has_nan_gpu.get_val(&has_nan);
  this->c->check();
  
  return (bool) has_nan;
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
REAL gpumat<REAL>::get(const len_t i) const
{
  this->check_index(i);

  REAL ret;
  this->c->mem_gpu2cpu(&ret, this->data + i, sizeof(REAL));
  return ret;
}

/**
  @brief Get the specified value.
  
  @param[in] i,j The indices of the desired value, 0-indexed.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename REAL>
REAL gpumat<REAL>::get(const len_t i, const len_t j) const
{
  this->check_index(i, j);

  REAL ret;
  this->c->mem_gpu2cpu(&ret, this->data + (i + this->m*j), sizeof(REAL));
  return ret;
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
void gpumat<REAL>::set(const len_t i, const REAL v)
{
  this->check_index(i);
  this->c->mem_cpu2gpu(this->data + i, &v, sizeof(REAL));
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i,j The indices of the desired value, 0-indexed.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
 */
template <typename REAL>
void gpumat<REAL>::set(const len_t i, const len_t j, const REAL v)
{
  this->check_index(i, j);
  this->c->mem_cpu2gpu(this->data + (i + this->m*j), &v, sizeof(REAL));
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
void gpumat<REAL>::get_row(const len_t i, gpuvec<REAL> &v) const
{
  if (i < 0 || i >= this->m)
    throw std::logic_error("invalid matrix row");
  
  v.resize(this->m);
  
  fml::kernelfuns::kernel_get_row<<<dim_grid, dim_block>>>(i, this->m, this->n, this->data, v.data_ptr());
  this->c->check();
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
void gpumat<REAL>::get_col(const len_t j, gpuvec<REAL> &v) const
{
  if (j < 0 || j >= this->n)
    throw std::logic_error("invalid matrix column");
  
  v.resize(this->n);
  
  fml::kernelfuns::kernel_get_col<<<dim_grid, dim_block>>>(j, this->m, this->n, this->data, v.data_ptr());
  this->c->check();
}



/**
  @brief See if the two objects are the same.
  
  @param[in] Comparison object.
  @return If the dimensions mismatch, then `false` is necessarily returned.
  Next, if the card objects have different ordinal IDs, then `false` is
  returned. Next, if the pointer to the internal storage arrays match, then
  `true` is returned. Otherwise the objects are compared value by value.
 */
template <typename T>
bool gpumat<T>::operator==(const gpumat<T> &x) const
{
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  else if (this->c->get_id() != x.get_card()->get_id())
    return false;
  else if (this->data == x.data_ptr())
    return true;
  
  int all_eq = 1;
  gpuscalar<int> all_eq_gpu(c, all_eq);
  
  fml::kernelfuns::kernel_all_eq<<<dim_grid, dim_block>>>(this->m, this->n, this->data, x.data_ptr(), all_eq_gpu.data_ptr());
  
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
bool gpumat<T>::operator!=(const gpumat<T> &x) const
{
  return !(*this == x);
}



/**
  @brief Operator that sets the LHS to a shallow copy of the input. Desctruction
  of the LHS object will not result in the internal array storage being freed.
  
  @param[in] x Setter value.
 */
template <typename REAL>
gpumat<REAL>& gpumat<REAL>::operator=(const gpumat<REAL> &x)
{
  this->c = x.get_card();
  
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
void gpumat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    this->c->mem_free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void gpumat<REAL>::check_params(len_t nrows, len_t ncols)
{
  if (nrows < 0 || ncols < 0)
    throw std::runtime_error("invalid dimensions");
}



template <typename REAL>
void gpumat<REAL>::check_gpu(std::shared_ptr<card> gpu)
{
  if (!gpu->valid_card())
    throw std::runtime_error("GPU card object is invalid");
}


#endif
