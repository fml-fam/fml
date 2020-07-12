// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_INTERNALS_PARMAT_H
#define FML_PAR_INTERNALS_PARMAT_H
#pragma once


#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

#include "../../_internals/rand.hh"
#include "../../_internals/types.hh"
#include "../comm.hh"


namespace fml
{
  template <class MAT, class VEC, typename REAL>
  class parmat
  {
    public:
      parmat(){};
      parmat(comm &mpi_comm, MAT &data_);
      parmat(parmat<MAT, VEC, REAL> &&x);
      
      void print(uint8_t ndigits=4, bool add_final_blank=true);
      void info();
      
      void fill_zero();
      void fill_val(const REAL v);
      void fill_linspace(const REAL start, const REAL stop);
      void fill_eye();
      void fill_diag(const VEC &d);
      void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
      void fill_runif(const REAL min=0, const REAL max=1);
      void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
      void fill_rnorm(const REAL mean=0, const REAL sd=1);
      
      // void diag(cpuvec<REAL> &v);
      // void antidiag(cpuvec<REAL> &v);
      void scale(const REAL s);
      // void rev_rows();
      void rev_cols();
      
      bool any_inf() const;
      bool any_nan() const;
      
      REAL get(const len_global_t i) const;
      REAL get(const len_global_t i, const len_t j) const;
      void set(const len_global_t i, const REAL v);
      void set(const len_global_t i, const len_t j, const REAL v);
      // void get_row(const len_global_t i, VEC &v) const;
      // void get_col(const len_global_t j, VEC &v) const;

      bool operator==(const parmat<MAT, VEC, REAL> &x) const;
      bool operator!=(const parmat<MAT, VEC, REAL> &x) const;
      parmat<MAT, VEC, REAL>& operator=(const parmat<MAT, VEC, REAL> &x);
      
      len_global_t nrows() const {return m_global;};
      len_local_t nrows_local() const {return data.nrows();};
      len_local_t ncols() const {return data.ncols();};
      len_global_t nrows_before() const {return nb4;};
      comm get_comm() const {return r;};
      const MAT& data_obj() const {return data;};
      MAT& data_obj() {return data;};
      
    protected:
      MAT data;
      len_global_t m_global;
      comm r;
      len_global_t nb4;
      
      void num_preceding_rows();
      len_t get_local_dim();
      void check_index(const len_global_t i) const;
      void check_index(const len_global_t i, const len_t j) const;
  };
}



template <class MAT, class VEC, typename REAL>
fml::parmat<MAT, VEC, REAL>::parmat(fml::comm &mpi_comm, MAT &data_)
{
  r = mpi_comm;
  data = data_;
  
  m_global = (len_global_t) data.nrows();
  r.allreduce(1, &(m_global));
  num_preceding_rows();
}



template <class MAT, class VEC, typename REAL>
fml::parmat<MAT, VEC, REAL>::parmat(fml::parmat<MAT, VEC, REAL> &&x)
{
  this->data = x.data_obj();
  this->m_global = x.nrows();
  this->r = x.get_comm();
  this->nb4 = x.nrows_before();
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::info()
{
  r.printf(0, "# parmat");
  r.printf(0, " %" PRIu64 "x%d", m_global, data.ncols());
  r.printf(0, " type=%s", typeid(REAL).name());
  r.printf(0, "\n");
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::fill_zero()
{
  data.fill_zero();
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::fill_val(const REAL v)
{
  data.fill_val(v);
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::fill_runif(const uint32_t seed, const REAL min, const REAL max)
{
  data.fill_runif(seed, min, max);
}

template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::fill_runif(const REAL min, const REAL max)
{
  uint32_t seed = fml::rand::get_seed() + r.rank();
  data.fill_runif(seed, min, max);
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::fill_rnorm(const uint32_t seed, const REAL mean, const REAL sd)
{
  data.fill_rnorm(seed, mean, sd);
}

template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::fill_rnorm(const REAL mean, const REAL sd)
{
  uint32_t seed = fml::rand::get_seed() + r.rank();
  data.fill_rnorm(seed, mean, sd);
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::scale(const REAL v)
{
  data.scale(v);
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::rev_cols()
{
  data.rev_cols();
}



template <class MAT, class VEC, typename REAL>
bool fml::parmat<MAT, VEC, REAL>::any_inf() const
{
  int ret = (int) data.any_inf();
  r.allreduce(1, &ret);
  return (bool) ret;
}



template <class MAT, class VEC, typename REAL>
bool fml::parmat<MAT, VEC, REAL>::any_nan() const
{
  int ret = (int) data.any_nan();
  r.allreduce(1, &ret);
  return (bool) ret;
}



template <class MAT, class VEC, typename REAL>
REAL fml::parmat<MAT, VEC, REAL>::get(const len_global_t i) const
{
  check_index(i);
  
  len_global_t row = i % m_global;
  len_t j = (len_t) (i / m_global);
  
  REAL ret;
  if (row >= nb4 && row < nb4+data.nrows())
    ret = data.get(row-nb4, j);
  else
    ret = 0;
  
  r.allreduce(1, &ret);
  return ret;
}

template <class MAT, class VEC, typename REAL>
REAL fml::parmat<MAT, VEC, REAL>::get(const len_global_t i, const len_t j) const
{
  check_index(i, j);
  
  REAL ret;
  if (i >= nb4 && i < nb4+data.nrows())
    ret = data.get(i-nb4, j);
  else
    ret = 0;
  
  r.allreduce(1, &ret);
  return ret;
}

template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::set(const len_global_t i, const REAL v)
{
  check_index(i);
  
  len_global_t row = i % m_global;
  len_t j = (len_t) (i / m_global);
  
  if (row >= nb4 && row < nb4+data.nrows())
    data.set(row-nb4, j, v);
}

template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::set(const len_global_t i, const len_t j, const REAL v)
{
  check_index(i, j);
  
  if (i >= nb4 && i < nb4+data.nrows())
    data.set(i-nb4, j, v);
}



template <class MAT, class VEC, typename REAL>
bool fml::parmat<MAT, VEC, REAL>::operator==(const fml::parmat<MAT, VEC, REAL> &x) const
{
  int neq_count = (int) (data != x.data_obj());
  r.allreduce(1, &neq_count);
  
  return (neq_count == 0);
}

template <class MAT, class VEC, typename REAL>
bool fml::parmat<MAT, VEC, REAL>::operator!=(const fml::parmat<MAT, VEC, REAL> &x) const
{
  return !(*this == x);
}



template <class MAT, class VEC, typename REAL>
fml::parmat<MAT, VEC, REAL>& fml::parmat<MAT, VEC, REAL>::operator=(const fml::parmat<MAT, VEC, REAL> &x)
{
  this->data = x.data_obj();
  this->m_global = x.nrows();
  this->r = x.get_comm();
  this->nb4 = x.nrows_before();
}


// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::num_preceding_rows()
{
  int myrank = r.rank();
  int size = r.size();;
  
  nb4 = 0;
  len_t m_local = data.nrows();
  
  for (int rank=1; rank<size; rank++)
  {
    if (myrank == (rank - 1))
    {
      len_global_t nb4_send = nb4 + ((len_global_t) m_local);
      r.send(1, &nb4_send, rank);
    }
    else if (myrank == rank)
    {
      len_global_t nr_prev_rank;
      r.recv(1, &nr_prev_rank, rank-1);
      
      nb4 += nr_prev_rank;
    }
  }
}



template <class MAT, class VEC, typename REAL>
len_t fml::parmat<MAT, VEC, REAL>::get_local_dim()
{
  len_t local = m_global / r.size();
  len_t rem = (len_t) (m_global - (len_global_t) local*r.size());
  if (r.rank()+1 <= rem)
    local++;
  
  return local;
}



template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::check_index(const len_global_t i) const
{
  if (i < 0 || i >= (m_global * data.ncols()))
    throw std::runtime_error("index out of bounds");
}

template <class MAT, class VEC, typename REAL>
void fml::parmat<MAT, VEC, REAL>::check_index(const len_global_t i, const len_t j) const
{
  if (i < 0 || i >= m_global || j < 0 || j >= data.ncols())
    throw std::runtime_error("index out of bounds");
}


#endif
