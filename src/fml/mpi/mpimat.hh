// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_MPIMAT_H
#define FML_MPI_MPIMAT_H
#pragma once


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "grid.hh"
#include "internals/bcutils.hh"

#include "../_internals/arraytools/src/arraytools.hpp"

#include "../_internals/print.hh"
#include "../_internals/rand.hh"
#include "../_internals/omp.hh"
#include "../_internals/types.hh"
#include "../_internals/unimat.hh"

#include "../cpu/cpuvec.hh"


namespace fml
{
  /**
   * @brief Matrix class for data distributed over MPI in the 2-d block cyclic
      format. 
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  class mpimat : public fml::unimat<REAL>
  {
    public:
      mpimat(const grid &blacs_grid);
      mpimat(const grid &blacs_grid, int bf_rows, int bf_cols);
      mpimat(const grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols);
      mpimat(const grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct=false);
      mpimat(const mpimat &x);
      ~mpimat();
      
      void resize(len_t nrows, len_t ncols);
      void resize(len_t nrows, len_t ncols, int bf_rows, int bf_cols);
      void inherit(grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct=false);
      mpimat<REAL> dupe() const;
      
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
      void get_col(const len_t j, cpuvec<REAL> &v) const;
      
      bool operator==(const mpimat<REAL> &x) const;
      bool operator!=(const mpimat<REAL> &x) const;
      mpimat<REAL>& operator=(const mpimat<REAL> &x);
      
      len_local_t nrows_local() const {return m_local;};
      len_local_t ncols_local() const {return n_local;};
      int bf_rows() const {return mb;};
      int bf_cols() const {return nb;};
      int* desc_ptr() {return desc;};
      const int* desc_ptr() const {return desc;};
      const grid get_grid() const {return g;};
      
    protected:
      len_local_t m_local;
      len_local_t n_local;
      int mb;
      int nb;
      int desc[9];
      grid g;
      
    private:
      void free();
      void check_params(len_t nrows, len_t ncols, int bf_rows, int bf_cols);
      void check_grid(const grid &blacs_grid);
      REAL get_val_from_global_index(len_t gi, len_t gj) const;
  };
}



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
  @brief Construct matrix object with no internal allocated storage.
  
  @param[in] blacs_grid Scalapack grid object.
  
  @comm The method has no communication.
  
  @code
  grid g = grid(PROC_GRID_SQUARE);
  mpimat<float> x(g);
  @endcode
 */
template <typename REAL>
fml::mpimat<REAL>::mpimat(const fml::grid &blacs_grid)
{
  check_grid(blacs_grid);
  
  this->m = 0;
  this->n = 0;
  this->m_local = 0;
  this->n_local = 0;
  this->data = NULL;
  
  this->mb = 0;
  this->nb = 0;
  
  this->g = blacs_grid;
  
  this->free_data = true;
}



/**
  @brief Construct matrix object.
  
  @param[in] blacs_grid Scalapack grid object.
  @param[in] nrows,ncols Number rows/columns of the matrix.
  @param[in] bf_rows,bf_cols Row/column blocking factor.
  
  @except If the allocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
  
  @comm The method has no communication.
  
  @code
  grid g = grid(PROC_GRID_SQUARE);
  mpimat<float> x(g, 1, 1);
  @endcode
 */
template <typename REAL>
fml::mpimat<REAL>::mpimat(const fml::grid &blacs_grid, int bf_rows, int bf_cols)
{
  check_grid(blacs_grid);
  
  this->m = 0;
  this->n = 0;
  this->m_local = 0;
  this->n_local = 0;
  this->data = NULL;
  
  this->mb = bf_rows;
  this->nb = bf_cols;
  
  this->g = blacs_grid;
  
  this->free_data = true;
}



/**
  @brief Construct matrix object.
  
  @param[in] blacs_grid Scalapack grid object.
  @param[in] nrows,ncols Number rows/columns of the array, i.e. the length of
  the array is nrows*ncols.
  @param[in] bf_rows,bf_cols Row/column blocking factor.
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
  
  @comm The method has no communication.
  
  @code
  grid g = grid(PROC_GRID_SQUARE);
  mpimat<float> x(g, 3, 2, 1, 1);
  @endcode
 */
template <typename REAL>
fml::mpimat<REAL>::mpimat(const fml::grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols)
{
  check_params(nrows, ncols, bf_rows, bf_cols);
  check_grid(blacs_grid);
  
  this->m_local = fml::bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  this->n_local = fml::bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  
  fml::bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, this->m_local);
  
  const size_t len = (size_t) this->m_local * this->n_local * sizeof(REAL);
  this->data = (REAL*) std::malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
  
  this->m = nrows;
  this->n = ncols;
  this->mb = bf_rows;
  this->nb = bf_cols;
  this->g = blacs_grid;
  
  this->free_data = true;
}



/**
  @brief Construct matrix object with inherited data. Essentially the same as
  using the minimal constructor and immediately calling the `inherit()` method.
  
  @param[in] blacs_grid Scalapack grid object.
  @param[in] data_ Storage array.
  @param[in] nrows,ncols Number rows/columns of the array, i.e. the length of
  the array is nrows*ncols.
  @param[in] bf_rows,bf_cols Row/column blocking factor.
  @param[in] free_on_destruct Should the inherited array `data_` be freed when
  the matrix object is destroyed?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
  
  @comm The method has no communication.
 */
template <typename REAL>
fml::mpimat<REAL>::mpimat(const fml::grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct)
{
  check_params(nrows, ncols, bf_rows, bf_cols);
  check_grid(blacs_grid);
  
  this->m_local = fml::bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  this->n_local = fml::bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  
  fml::bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, this->m_local);
  
  this->m = nrows;
  this->n = ncols;
  this->mb = bf_rows;
  this->nb = bf_cols;
  this->g = blacs_grid;
  
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
fml::mpimat<REAL>::mpimat(const fml::mpimat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  
  this->m_local = x.nrows_local();
  this->n_local = x.ncols_local();
  this->mb = x.bf_rows();
  this->nb = x.bf_cols();
  
  memcpy(this->desc, x.desc_ptr(), 9*sizeof(int));
  
  fml::grid g = x.get_grid();
  this->g = g;
  
  this->data = x.data_ptr();
  
  this->free_data = false;
}



template <typename REAL>
fml::mpimat<REAL>::~mpimat()
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
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::resize(len_t nrows, len_t ncols)
{
  check_params(nrows, ncols, this->mb, this->nb);
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  const size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
  if (len == oldlen)
  {
    this->m = nrows;
    this->n = ncols;
    
    this->m_local = fml::bcutils::numroc(nrows, this->mb, this->g.myrow(), 0, this->g.nprow());
    this->n_local = fml::bcutils::numroc(ncols, this->nb, this->g.mycol(), 0, this->g.npcol());
    
    fml::bcutils::descinit(this->desc, this->g.ictxt(), nrows, ncols, this->mb, this->nb, this->m_local);
    
    return;
  }
  
  this->m_local = fml::bcutils::numroc(nrows, this->mb, this->g.myrow(), 0, this->g.nprow());
  this->n_local = fml::bcutils::numroc(ncols, this->nb, this->g.mycol(), 0, this->g.npcol());
  
  void *realloc_ptr;
  if (oldlen == 0)
    realloc_ptr = malloc(len);
  else
    realloc_ptr = realloc(this->data, len);
  
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (REAL*) realloc_ptr;
  
  fml::bcutils::descinit(this->desc, this->g.ictxt(), nrows, ncols, this->mb, this->nb, this->m_local);
  
  this->m = nrows;
  this->n = ncols;
}



/**
  @brief Resize the internal object storage.
  
  @param[in] nrows,ncols Number rows/columns needed.
  @param[in] bf_rows,bf_cols Row/column blocking factor.
  
  @allocs Resizing triggers a re-allocation.
  
  @except If the reallocation fails, a `bad_alloc` exception will be thrown.
  If the input values are invalid, a `runtime_error` exception will be thrown.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::resize(len_t nrows, len_t ncols, int bf_rows, int bf_cols)
{
  check_params(nrows, ncols, bf_rows, bf_cols);
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  const size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
  if (len == oldlen && this->mb == bf_rows && this->nb == bf_cols)
  {
    this->m = nrows;
    this->n = ncols;
    
    this->m_local = fml::bcutils::numroc(nrows, bf_rows, this->g.myrow(), 0, this->g.nprow());
    this->n_local = fml::bcutils::numroc(ncols, bf_cols, this->g.mycol(), 0, this->g.npcol());
    
    fml::bcutils::descinit(this->desc, this->g.ictxt(), nrows, ncols, bf_rows, bf_cols, this->m_local);
    
    return;
  }
  
  this->mb = bf_rows;
  this->nb = bf_cols;
  
  this->m_local = fml::bcutils::numroc(nrows, this->mb, this->g.myrow(), 0, this->g.nprow());
  this->n_local = fml::bcutils::numroc(ncols, this->nb, this->g.mycol(), 0, this->g.npcol());
  
  void *realloc_ptr;
  if (oldlen == 0)
    realloc_ptr = malloc(len);
  else
    realloc_ptr = realloc(this->data, len);
  
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (REAL*) realloc_ptr;
  
  fml::bcutils::descinit(this->desc, this->g.ictxt(), nrows, ncols, this->mb, this->nb, this->m_local);
  
  this->m = nrows;
  this->n = ncols;
}



/**
  @brief Set the internal object storage to the specified array.
  
  @param[in] data Value storage.
  @param[in] nrows,ncols Number rows/columns of the matrix. The product of
  these should be the length of the input `data`.
  @param[in] bf_rows,bf_cols Row/column blocking factor.
  @param[in] free_on_destruct Should the object destructor free the internal
  array `data`?
  
  @except If the input values are invalid, a `runtime_error` exception will be
  thrown.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::inherit(fml::grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct)
{
  check_params(nrows, ncols, bf_rows, bf_cols);
  check_grid(blacs_grid);
  
  this->free();
  
  m_local = fml::bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  n_local = fml::bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  fml::bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, m_local);
  
  this->m = nrows;
  this->n = ncols;
  this->mb = bf_rows;
  this->nb = bf_cols;
  this->g = blacs_grid;
  
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



/// @brief Duplicate the object in a deep copy.
template <typename REAL>
fml::mpimat<REAL> fml::mpimat<REAL>::dupe() const
{
  fml::mpimat<REAL> dup(this->g, this->m, this->n, this->mb, this->nb);
  
  const size_t len = (size_t) this->m_local * this->n_local * sizeof(REAL);
  
  memcpy(dup.data_ptr(), this->data, len);
  memcpy(dup.desc_ptr(), this->desc, 9*sizeof(int));
  
  return dup;
}



// printers

/**
  @brief Print all values in the object.
  
  @details Printing will only be done by rank 0.
  
  @param[in] ndigits Number of decimal digits to print.
  @param[in] add_final_blank Should a final blank line be printed?
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::print(uint8_t ndigits, bool add_final_blank) const
{
  for (len_t gi=0; gi<this->m; gi++)
  {
    for (len_t gj=0; gj<this->n; gj++)
    {
      const int pr = fml::bcutils::g2p(gi, this->mb, this->g.nprow());
      const int pc = fml::bcutils::g2p(gj, this->nb, this->g.npcol());
      
      const int i = fml::bcutils::g2l(gi, this->mb, this->g.nprow());
      const int j = fml::bcutils::g2l(gj, this->nb, this->g.npcol());
      
      REAL d;
      if (this->g.rank0())
      {
        if (pr == 0 && pc == 0)
          d = this->data[i + this->m_local*j];
        else
          this->g.recv(1, 1, &d, pr, pc);
        
        this->printval(d, ndigits);
      }
      else if (pr == this->g.myrow() && pc == this->g.mycol())
      {
        d = this->data[i + this->m_local*j];
        this->g.send(1, 1, &d, 0, 0);
      }
    }
    
    this->g.printf(0, 0, "\n");
  }
  
  if (add_final_blank)
    this->g.printf(0, 0, "\n");
}



/**
  @brief Print some brief information about the object.
  
  @details Printing will only be done by rank 0.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::info() const
{
  if (this->g.rank0())
  {
    fml::print::printf("# mpimat");
    fml::print::printf(" %dx%d", this->m, this->n);
    fml::print::printf(" with %dx%d blocking", this->mb, this->nb);
    fml::print::printf(" on %dx%d grid", this->g.nprow(), this->g.npcol());
    fml::print::printf(" type=%s", typeid(REAL).name());
    fml::print::printf("\n");
  }
}



// fillers

/**
  @brief Set all values to zero.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::fill_zero()
{
  const size_t len = (size_t) m_local * n_local * sizeof(REAL);
  memset(this->data, 0, len);
}



/**
  @brief Set all values to input value.
  
  @param[in] v Value to set all data values to.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::fill_val(const REAL v)
{
  #pragma omp parallel for if((this->m_local)*(this->n_local) > fml::omp::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n_local; j++)
  {
    #pragma omp simd
    for (len_t i=0; i<this->m_local; i++)
      this->data[i + this->m_local*j] = v;
  }
}



/**
  @brief Set values to linearly spaced numbers.
  
  @param[in] start,stop Beginning/ending numbers. If not supplied, the matrix
  will be filled with whole numbers from 1 to the total number of elements.
  
  @comm The method has no communication.
 */
template <typename T>
void fml::mpimat<T>::fill_linspace()
{
  T start = 1;
  T stop = (T) (this->m * this->n);
  this->fill_linspace(start, stop);
}

template <typename REAL>
void fml::mpimat<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const REAL v = (stop-start)/((REAL) this->m*this->n - 1);
    
    #pragma omp parallel for if((this->m_local)*(this->n_local) > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<this->n_local; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<this->m_local; i++)
      {
        const int gi = fml::bcutils::l2g(i, this->mb, this->g.nprow(), this->g.myrow());
        const int gj = fml::bcutils::l2g(j, this->nb, this->g.npcol(), this->g.mycol());
        
        this->data[i + this->m_local*j] = v*((REAL) gi + this->m*gj) + start;
      }
    }
  }
}

template <>
inline void fml::mpimat<int>::fill_linspace(const int start, const int stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const float v = (stop-start)/((float) this->m*this->n - 1);
    
    #pragma omp parallel for if((this->m_local)*(this->n_local) > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<this->n_local; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<this->m_local; i++)
      {
        const int gi = fml::bcutils::l2g(i, this->mb, this->g.nprow(), this->g.myrow());
        const int gj = fml::bcutils::l2g(j, this->nb, this->g.npcol(), this->g.mycol());
        
        this->data[i + this->m_local*j] = (int) roundf(v*((float) gi + this->m*gj) + start);
      }
    }
  }
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::fill_eye()
{
  fml::cpuvec<REAL> v(1);
  v.set(0, 1);
  this->fill_diag(v);
}



/**
  @brief Set diagonal entries of the matrix to those in the vector.
  
  @details If the vector is smaller than the matrix diagonal, the vector will
  recycle until the matrix diagonal is filled. If the vector is longer, then
  not all of it will be used.
  
  @param[in] v Vector of values to set the matrix diagonal to.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::fill_diag(const fml::cpuvec<REAL> &v)
{
  REAL *v_d = v.data_ptr();
  
  #pragma omp parallel for if((this->m_local)*(this->n_local) > fml::omp::OMP_MIN_SIZE)
  for (len_local_t j=0; j<n_local; j++)
  {
    for (len_local_t i=0; i<m_local; i++)
    {
      const int gi = fml::bcutils::l2g(i, this->mb, this->g.nprow(), this->g.myrow());
      const int gj = fml::bcutils::l2g(j, this->nb, this->g.npcol(), this->g.mycol());
      
      if (gi == gj)
        this->data[i + this->m_local*j] = v_d[gi % v.size()];
      else
        this->data[i + this->m_local*j] = 0;
    }
  }
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @param[in] seed Seed for the rng.
  @param[in] min,max Parameters for the generator.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::fill_runif(const uint32_t seed, const REAL min, const REAL max)
{
  std::mt19937 mt(seed + g.myrow() + g.nprow()*g.mycol());
  static std::uniform_real_distribution<REAL> dist(min, max);
  
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
      this->data[i + this->m_local*j] = dist(mt);
  }
}

/// \overload
template <typename REAL>
void fml::mpimat<REAL>::fill_runif(const REAL min, const REAL max)
{
  uint32_t seed = fml::rand::get_seed() + (g.myrow() + g.nprow()*g.mycol());
  this->fill_runif(seed, min, max);
}



/**
  @brief Set diagonal entries to 1 and non-diagonal entries to 0.
  
  @param[in] seed Seed for the rng.
  @param[in] mean,sd Parameters for the generator.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::fill_rnorm(const uint32_t seed, const REAL mean, const REAL sd)
{
  std::mt19937 mt(seed + g.myrow() + g.nprow()*g.mycol());
  static std::normal_distribution<REAL> dist(mean, sd);
  
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
      this->data[i + this->m_local*j] = dist(mt);
  }
}

/// \overload
template <typename REAL>
void fml::mpimat<REAL>::fill_rnorm(const REAL mean, const REAL sd)
{
  uint32_t seed = fml::rand::get_seed() + (g.myrow() + g.nprow()*g.mycol());
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
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::diag(fml::cpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  v.fill_zero();
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp parallel for if(minmn > fml::omp::OMP_MIN_SIZE)
  for (len_t gi=0; gi<minmn; gi++)
  {
    const len_local_t i = fml::bcutils::g2l(gi, this->mb, this->g.nprow());
    const len_local_t j = fml::bcutils::g2l(gi, this->nb, this->g.npcol());
    
    const int pr = fml::bcutils::g2p(gi, this->mb, this->g.nprow());
    const int pc = fml::bcutils::g2p(gi, this->nb, this->g.npcol());
    
    if (pr == this->g.myrow() && pc == this->g.mycol())
      v_ptr[gi] = this->data[i + this->m_local*j];
  }
  
  
  this->g.allreduce(minmn, 1, v_ptr, 'A');
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
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::antidiag(fml::cpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  v.fill_zero();
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp parallel for if(minmn > fml::omp::OMP_MIN_SIZE)
  for (len_t gi=0; gi<minmn; gi++)
  {
    const len_local_t i = fml::bcutils::g2l(this->m-1-gi, this->mb, this->g.nprow());
    const len_local_t j = fml::bcutils::g2l(gi, this->nb, this->g.npcol());
    
    const int pr = fml::bcutils::g2p(this->m-1-gi, this->mb, this->g.nprow());
    const int pc = fml::bcutils::g2p(gi, this->nb, this->g.npcol());
    
    if (pr == this->g.myrow() && pc == this->g.mycol())
      v_ptr[gi] = this->data[i + this->m_local*j];
  }
  
  
  this->g.allreduce(minmn, 1, v_ptr, 'A');
}



/**
  @brief Multiply all values by the input value.
  
  @param[in] s Scaling value.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::scale(const REAL s)
{
  #pragma omp parallel for if((this->m_local)*(this->n_local) > fml::omp::OMP_MIN_SIZE)
  for (len_local_t j=0; j<this->n_local; j++)
  {
    #pragma omp simd
    for (len_local_t i=0; i<this->m_local; i++)
      this->data[i + this->m_local*j] *= s;
  }
}



/**
  @brief Reverse the rows of the matrix.
  
  @comm The method will communicate with all rows in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::rev_rows()
{
  cpuvec<REAL> tmp(this->nb);
  REAL *tmp_d = tmp.data_ptr();
  
  const int myrow = this->g.myrow();
  const int mycol = this->g.mycol();
  
  for (len_t gj=0; gj<this->n; gj+=this->nb)
  {
    const len_t j = fml::bcutils::g2l(gj, this->nb, this->g.npcol());
    const int pc = fml::bcutils::g2p(gj, this->nb, this->g.npcol());
    
    for (len_t gi=0; gi<this->m/2; gi++)
    {
      const len_t i = fml::bcutils::g2l(gi, this->mb, this->g.nprow());
      const len_t gi_rev = this->m - gi - 1;
      
      const int pr = fml::bcutils::g2p(gi, this->mb, this->g.nprow());
      const int pr_rev = fml::bcutils::g2p(gi_rev, this->mb, this->g.nprow());
      
      if ((pr == myrow || pr_rev == myrow) && pc == mycol)
      {
        const len_t i_rev = fml::bcutils::g2l(gi_rev, this->mb, this->g.nprow());
        const len_t cplen = std::min(this->nb, this->n - gj);
        
        if (pr == pr_rev)
        {
          if (i != i_rev)
          {
            #pragma omp for simd
            for (len_t jj=0; jj<cplen; jj++)
              tmp_d[jj] = this->data[i + this->m_local*(j+jj)];
            
            #pragma omp for simd
            for (len_t jj=0; jj<cplen; jj++)
              this->data[i + this->m_local*(j+jj)] = this->data[i_rev + this->m_local*(j+jj)];
            
            #pragma omp for simd
            for (len_t jj=0; jj<cplen; jj++)
              this->data[i_rev + this->m_local*(j+jj)] = tmp_d[jj];
          }
        }
        else
        {
          // oroginal indexed process sends/recvs and higher recvs/sends
          if (pr == myrow)
          {
            len_t idx = i + this->m_local*j;
            this->g.send(1, cplen, this->m_local, this->data + idx, pr_rev, pc);
            this->g.recv(1, cplen, 1, tmp_d, pr_rev, pc);
            
            #pragma omp for simd
            for (len_t jj=0; jj<cplen; jj++)
              this->data[idx + this->m_local*jj] = tmp_d[jj];
          }
          else
          {
            len_t idx = i_rev + this->m_local*j;
            this->g.recv(1, cplen, 1, tmp_d, pr, pc);
            this->g.send(1, cplen, this->m_local, this->data + idx, pr, pc);
            
            #pragma omp for simd
            for (len_t jj=0; jj<cplen; jj++)
              this->data[idx + this->m_local*jj] = tmp_d[jj];
          }
        }
      }
      
      this->g.barrier('R');
    }
  }
}



/**
  @brief Reverse the columns of the matrix.
  
  @comm The method will communicate with all columns in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::rev_cols()
{
  cpuvec<REAL> tmp(this->mb);
  REAL *tmp_d = tmp.data_ptr();
  
  const int myrow = this->g.myrow();
  const int mycol = this->g.mycol();
  
  for (len_t gj=0; gj<this->n/2; gj++)
  {
    const len_t j = fml::bcutils::g2l(gj, this->nb, this->g.npcol());
    const len_t gj_rev = this->n - gj - 1;
    const len_t j_rev = fml::bcutils::g2l(gj_rev, this->nb, this->g.npcol());
    
    const int pc = fml::bcutils::g2p(gj, this->nb, this->g.npcol());
    const int pc_rev = fml::bcutils::g2p(gj_rev, this->nb, this->g.npcol());
    
    for (len_t gi=0; gi<this->m; gi+=this->mb)
    {
      const len_t i = fml::bcutils::g2l(gi, this->mb, this->g.nprow());
      const int pr = fml::bcutils::g2p(gi, this->mb, this->g.nprow());
      
      if (pr == myrow && (pc == mycol || pc_rev == mycol))
      {
        const len_t cplen = std::min(this->mb, this->m - gi);
        
        if (pc == pc_rev)
        {
          if (j != j_rev)
          {
            #pragma omp for simd
            for (len_t ii=0; ii<cplen; ii++)
              tmp_d[ii] = this->data[i+ii + this->m_local*j];
            
            #pragma omp for simd
            for (len_t ii=0; ii<cplen; ii++)
              this->data[i+ii + this->m_local*j] = this->data[i+ii + this->m_local*j_rev];
            
            #pragma omp for simd
            for (len_t ii=0; ii<cplen; ii++)
              this->data[i+ii + this->m_local*j_rev] = tmp_d[ii];
          }
        }
        else
        {
          // oroginal indexed process sends/recvs and higher recvs/sends
          if (pc == mycol)
          {
            len_t idx = i + this->m_local*j;
            this->g.send(cplen, 1, this->m_local, this->data + idx, pr, pc_rev);
            this->g.recv(cplen, 1, 1, tmp_d, pr, pc_rev);
            
            #pragma omp for simd
            for (len_t ii=0; ii<cplen; ii++)
              this->data[idx+ii] = tmp_d[ii];
          }
          else
          {
            len_t idx = i + this->m_local*j_rev;
            this->g.recv(cplen, 1, 1, tmp_d, pr, pc);
            this->g.send(cplen, 1, this->m_local, this->data + idx, pr, pc);
            
            #pragma omp for simd
            for (len_t ii=0; ii<cplen; ii++)
              this->data[idx+ii] = tmp_d[ii];
          }
        }
      }
      
      this->g.barrier('C');
    }
  }
}



/**
  @brief Are any values infinite?
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
bool fml::mpimat<REAL>::any_inf() const
{
  int found_inf = 0;
  for (len_local_t j=0; j<n_local; j++)
  {
    for (len_local_t i=0; i<m_local; i++)
    {
      if (isinf(this->data[i + this->m_local*j]))
      {
        found_inf = 1;
        break;
      }
    }
  }
  
  this->g.allreduce(1, 1, &found_inf, 'A');
  
  return ((bool) found_inf);
}



/**
  @brief Are any values NaN?
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
bool fml::mpimat<REAL>::any_nan() const
{
  int found_nan = 0;
  for (len_local_t j=0; j<n_local; j++)
  {
    for (len_local_t i=0; i<m_local; i++)
    {
      if (isnan(this->data[i + this->m_local*j]))
      {
        found_nan = 1;
        break;
      }
    }
  }
  
  this->g.allreduce(1, 1, &found_nan, 'A');
  
  return ((bool) found_nan);
}



// operators

/**
  @brief Get the specified value.
  
  @details All processes receive the return.
  
  @param[in] i The index of the desired value, 0-indexed. The numbering
  considers the internal storage as a 1-dimensional array.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
REAL fml::mpimat<REAL>::get(const len_t i) const
{
  this->check_index(i);
  
  int gi = i % this->m;
  int gj = i / this->m;
  
  REAL ret = this->get_val_from_global_index(gi, gj);
  return ret;
}

/**
  @brief Get the specified value.
  
  @details All processes receive the return.
  
  @param[in] i,j The indices of the desired value, 0-indexed.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
REAL fml::mpimat<REAL>::get(const len_t i, const len_t j) const
{
  this->check_index(i, j);
  
  REAL ret = this->get_val_from_global_index(i, j);
  return ret;
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i The index of the desired value, 0-indexed. The numbering
  considers the internal storage as a 1-dimensional array.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::set(const len_t i, const REAL v)
{
  this->check_index(i);
  
  int gi = i % this->m;
  int gj = i / this->m;
  
  int pr = fml::bcutils::g2p(gi, this->mb, this->g.nprow());
  int pc = fml::bcutils::g2p(gj, this->nb, this->g.npcol());
  
  int li = fml::bcutils::g2l(gi, this->mb, this->g.nprow());
  int lj = fml::bcutils::g2l(gj, this->nb, this->g.npcol());
  
  if (pr == this->g.myrow() && pc == this->g.mycol())
    this->data[li + (this->m_local)*lj] = v;
}

/**
  @brief Set the storage at the specified index with the provided value.
  
  @param[in] i,j The indices of the desired value, 0-indexed.
  @param[in] v Setter value.
  
  @except If indices are out of bounds, the method will throw a `runtime_error`
  exception.
  
  @comm The method has no communication.
 */
template <typename REAL>
void fml::mpimat<REAL>::set(const len_t i, const len_t j, const REAL v)
{
  this->check_index(i, j);
  
  int pr = fml::bcutils::g2p(i, this->mb, this->g.nprow());
  int pc = fml::bcutils::g2p(j, this->nb, this->g.npcol());
  
  int li = fml::bcutils::g2l(i, this->mb, this->g.nprow());
  int lj = fml::bcutils::g2l(j, this->nb, this->g.npcol());
  
  if (pr == this->g.myrow() && pc == this->g.mycol())
    this->data[li + (this->m_local)*lj] = v;
}



/**
  @brief Get the specified row.
  
  @details The return vector of row values is set on all processes.
  
  @param[in] i The desired row, 0-indexed.
  @param[out] v The row values.
  
  @allocs If the output dimension is inappropriately sized, it will
  automatically be re-allocated.
  
  @except If `i` is an inappropriate value (i.e. does not refer to a matrix
  row), then the method will throw a `logic_error` exception. If a reallocation
  is triggered and fails, a `bad_alloc` exception will be thrown.
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::get_row(const len_t i, fml::cpuvec<REAL> &v) const
{
  if (i < 0 || i >= this->m)
    throw std::logic_error("invalid matrix row");
  
  v.resize(this->n);
  v.fill_zero();
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp parallel for if(this->n > fml::omp::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
  {
    const len_local_t i_local = fml::bcutils::g2l(i, this->mb, this->g.nprow());
    const len_local_t j_local = fml::bcutils::g2l(j, this->nb, this->g.npcol());
    
    const int pr = fml::bcutils::g2p(i, this->mb, this->g.nprow());
    const int pc = fml::bcutils::g2p(j, this->nb, this->g.npcol());
    
    if (pr == this->g.myrow() && pc == this->g.mycol())
      v_ptr[j] = this->data[i_local + this->m_local*j_local];
  }
  
  
  this->g.allreduce(this->n, 1, v_ptr, 'A');
}



/**
  @brief Get the specified column.
  
  @details The return vector of column values is set on all processes.
  
  @param[in] j The desired column, 0-indexed.
  @param[out] v The column values.
  
  @allocs If the output dimension is inappropriately sized, it will
  automatically be re-allocated.
  
  @except If `j` is an inappropriate value (i.e. does not refer to a matrix
  column), then the method will throw a `logic_error` exception. If a
  reallocation is triggered and fails, a `bad_alloc` exception will be thrown.
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
void fml::mpimat<REAL>::get_col(const len_t j, fml::cpuvec<REAL> &v) const
{
  if (j < 0 || j >= this->n)
    throw std::logic_error("invalid matrix column");
  
  v.resize(this->m);
  v.fill_zero();
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp parallel for if(this->m > fml::omp::OMP_MIN_SIZE)
  for (len_t i=0; i<this->m; i++)
  {
    const len_local_t i_local = fml::bcutils::g2l(i, this->mb, this->g.nprow());
    const len_local_t j_local = fml::bcutils::g2l(j, this->nb, this->g.npcol());
    
    const int pr = fml::bcutils::g2p(i, this->mb, this->g.nprow());
    const int pc = fml::bcutils::g2p(j, this->nb, this->g.npcol());
    
    if (pr == this->g.myrow() && pc == this->g.mycol())
      v_ptr[i] = this->data[i_local + this->m_local*j_local];
  }
  
  
  this->g.allreduce(this->m, 1, v_ptr, 'A');
}



/**
  @brief See if the two objects are the same.
  
  @details All processes receive the return.
  
  @param[in] Comparison object.
  
  @return If the dimensions mismatch, then `false` is necessarily returned.
  Next, if the grid objects have different ordinal context numbers, then `false`
  is returned. Next, if the pointer to the internal storage arrays match, then
  `true` is returned. Otherwise the objects are compared value by value.
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
bool fml::mpimat<REAL>::operator==(const fml::mpimat<REAL> &x) const
{
  // same dim, same blocking, same grid
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  else if (this->mb != x.bf_rows() || this->nb != x.bf_cols())
    return false;
  else if (this->g.ictxt() != x.g.ictxt())
    return false;
  
  const REAL *x_d = x.data_ptr();
  if (this->data == x_d)
    return true;
  
  int negation_ret = 0;
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
    {
      const REAL a = this->data[i + this->m_local*j];
      const REAL b = x_d[i + this->m_local*j];
      if (!arraytools::fltcmp::eq(a, b))
      {
        negation_ret = 1;
        break;
      }
    }
  }
  
  g.allreduce(1, 1, &negation_ret, 'A');
  
  return !((bool) negation_ret);
}

/**
  @brief See if the two objects are not the same. Uses same internal logic as
  the `==` method.
  
  @param[in] Comparison object.
  
  @comm The method will communicate across all processes in the BLACS grid.
 */
template <typename REAL>
bool fml::mpimat<REAL>::operator!=(const fml::mpimat<REAL> &x) const
{
  return !(*this == x);
}



/**
  @brief Operator that sets the LHS to a shallow copy of the input. Desctruction
  of the LHS object will not result in the internal array storage being freed.
  
  @param[in] x Setter value.
  
  @comm The method has no communication.
 */
template <typename REAL>
fml::mpimat<REAL>& fml::mpimat<REAL>::operator=(const fml::mpimat<REAL> &x)
{
  this->g = x.get_grid();
  
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->m_local = x.nrows_local();
  this->n_local = x.ncols_local();
  this->mb = x.bf_rows();
  this->nb = x.bf_cols();
  
  memcpy(this->desc, x.desc_ptr(), 9*sizeof(int));
  
  this->free_data = false;
  return *this;
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename REAL>
void fml::mpimat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void fml::mpimat<REAL>::check_params(len_t nrows, len_t ncols, int bf_rows, int bf_cols)
{
  if (nrows < 0 || ncols < 0)
    throw std::runtime_error("invalid dimensions");
  
  if (bf_rows <= 0 || bf_cols <= 0)
    throw std::runtime_error("invalid blocking factor");
}



template <typename REAL>
void fml::mpimat<REAL>::check_grid(const fml::grid &blacs_grid)
{
  if (!blacs_grid.valid_grid())
    throw std::runtime_error("invalid blacs grid");
}



template <typename REAL>
REAL fml::mpimat<REAL>::get_val_from_global_index(len_t gi, len_t gj) const
{
  REAL ret;
  
  int pr = fml::bcutils::g2p(gi, this->mb, this->g.nprow());
  int pc = fml::bcutils::g2p(gj, this->nb, this->g.npcol());
  
  int li = fml::bcutils::g2l(gi, this->mb, this->g.nprow());
  int lj = fml::bcutils::g2l(gj, this->nb, this->g.npcol());
  
  if (pr == this->g.myrow() && pc == this->g.mycol())
    ret = this->data[li + (this->m_local)*lj];
  else
    ret = (REAL) 0;
  
  g.allreduce(1, 1, &ret, 'A');
  
  return ret;
}


#endif
