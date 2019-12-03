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

#include "../../_internals/types.hh"
#include "../comm.hh"


template <class MAT, class VEC, typename REAL>
class parmat
{
  public:
    parmat(){};
    parmat(comm &mpi_comm, MAT &_data);
    
    void print(uint8_t ndigits=4, bool add_final_blank=true);
    void info();
    
    void fill_zero();
    void fill_one();
    void fill_val(const REAL v);
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_diag(const VEC &d);
    // void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
    // void fill_runif(const REAL min=0, const REAL max=1);
    // void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
    // void fill_rnorm(const REAL mean=0, const REAL sd=1);
    
    // void diag(cpuvec<REAL> &v);
    // void antidiag(cpuvec<REAL> &v);
    void scale(const REAL s);
    // void rev_rows();
    void rev_cols();
    
    bool any_inf() const;
    bool any_nan() const;
    
    len_global_t nrows() const {return m_global;};
    len_local_t nrows_local() const {return data.nrows();};
    len_local_t ncols() const {return data.ncols();};
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
};



template <class MAT, class VEC, typename REAL>
parmat<MAT, VEC, REAL>::parmat(comm &mpi_comm, MAT &_data)
{
  r = mpi_comm;
  data = _data;
  
  m_global = (len_global_t) data.nrows();
  r.allreduce(1, &(m_global));
  num_preceding_rows();
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::print(uint8_t ndigits, bool add_final_blank)
{
  len_t n = data.ncols();
  VEC pv(n);
  
  int myrank = r.rank();
  if (myrank == 0)
    data.print(ndigits, false);
  
  for (int rank=1; rank<r.size(); rank++)
  {
    if (rank == myrank)
    {
      len_t m = data.nrows();
      r.send(1, &m, 0);
      
      for (int i=0; i<m; i++)
      {
        data.get_row(i, pv);
        r.send(n, pv.data_ptr(), 0);
      }
    }
    else if (myrank == 0)
    {
      len_t m;
      r.recv(1, &m, rank);
      
      for (int i=0; i<m; i++)
      {
        r.recv(n, pv.data_ptr(), rank);
        pv.print(ndigits, false);
      }
    }
  
    r.barrier();
  }
  
  if (add_final_blank)
  {
    r.printf(0, "\n");
    r.barrier();
  }
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::info()
{
  r.printf(0, "# parmat");
  r.printf(0, " %" PRIu64 "x%d", m_global, data.ncols());
  r.printf(0, " type=%s", typeid(REAL).name());
  r.printf(0, "\n");
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::fill_zero()
{
  data.fill_zero();
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::fill_one()
{
  data.fill_one();
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::fill_val(const REAL v)
{
  data.fill_val(v);
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::scale(const REAL v)
{
  data.scale(v);
}



template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::rev_cols()
{
  data.rev_cols();
}



template <class MAT, class VEC, typename REAL>
bool parmat<MAT, VEC, REAL>::any_inf() const
{
  int ret = (int) data.any_inf();
  r.allreduce(1, &ret);
  return (bool) ret;
}



template <class MAT, class VEC, typename REAL>
bool parmat<MAT, VEC, REAL>::any_nan() const
{
  int ret = (int) data.any_nan();
  r.allreduce(1, &ret);
  return (bool) ret;
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <class MAT, class VEC, typename REAL>
void parmat<MAT, VEC, REAL>::num_preceding_rows()
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
len_t parmat<MAT, VEC, REAL>::get_local_dim()
{
  len_t local = m_global / r.size();
  len_t rem = (len_t) (m_global - (len_global_t) local*r.size());
  if (r.rank()+1 <= rem)
    local++;
  
  return local;
}


#endif
