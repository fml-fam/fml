// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_MPIHELPERS_H
#define FML_MPI_MPIHELPERS_H
#pragma once


#include <stdexcept>

#include "../_internals/arraytools/src/arraytools.hpp"

#include "../cpu/cpumat.hh"
#include "../cpu/cpuvec.hh"

#include "internals/bcutils.hh"
#include "grid.hh"
#include "mpimat.hh"


namespace fml
{
/// @brief MPI class helpers.
namespace mpihelpers
{
  /**
    @brief Copy data from an MPI object to a CPU object.
    
    @details Every process in the grid receives a copy of the data.
    
    @param[in] mpi Input data.
    @param[out] cpu Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void mpi2cpu_all(const mpimat<REAL_IN> &mpi, cpumat<REAL_OUT> &cpu)
  {
    grid g = mpi.get_grid();
    if (!g.ingrid())
      return;
    
    const len_local_t m_local = mpi.nrows_local();
    const len_local_t n_local = mpi.ncols_local();
    
    const int mb = mpi.bf_rows();
    
    const len_t m = mpi.nrows();
    const len_t n = mpi.ncols();
    
    if (m != cpu.nrows() || n != cpu.ncols())
      cpu.resize(m, n);
    
    cpu.fill_zero();
    
    REAL_OUT *gbl = cpu.data_ptr();
    const REAL_IN *sub = mpi.data_ptr();
    
    if (m_local > 0 && n_local > 0)
    {
      for (len_local_t j=0; j<n_local; j++)
      {
        const int gj = fml::bcutils::l2g(j, mpi.bf_cols(), g.npcol(), g.mycol());
        
        for (len_local_t i=0; i<m_local; i+=mb)
        {
          const int gi = fml::bcutils::l2g(i, mpi.bf_rows(), g.nprow(), g.myrow());
          
          for (int ii=0; ii<mb && ii+i<m_local; ii++)
            gbl[gi+ii + m*gj] = (REAL_OUT) sub[i+ii + m_local*j];
        }
      }
    }
    
    g.allreduce(m, n, gbl);
  }
  
  
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> mpi2cpu_all(const mpimat<REAL> &mpi)
  {
    cpumat<REAL> cpu;
    mpi2cpu_all(mpi, cpu);
    
    return cpu;
  }
  
  
  
  /**
    @brief Copy data from an MPI object to a CPU object.
    
    @details The process at grid positiong (rdest, cdest) will receive the full
    matrix.
    
    @param[in] mpi Input data.
    @param[out] cpu Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    @param[in] rdest,cdest Row/column position in the communicator grid of the
    receiving process.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated. Each process also needs
    temporary storage of the size `mpi.bf_rows()`.
    
    @except If an allocation or reallocation is triggered and fails, a
    `bad_alloc` exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void mpi2cpu(const mpimat<REAL_IN> &mpi, cpumat<REAL_OUT> &cpu, int rdest=0, int cdest=0)
  {
    const grid g = mpi.get_grid();
    if (!g.ingrid())
      return;
    
    bool i_am_ret = (g.myrow() == rdest && g.mycol() == cdest) ? true : false;
    
    const len_local_t m_local = mpi.nrows_local();
    
    const int mb = mpi.bf_rows();
    const int nb = mpi.bf_cols();
    
    const len_t m = mpi.nrows();
    const len_t n = mpi.ncols();
    
    if (i_am_ret)
    {
      if (m != cpu.nrows() || n != cpu.ncols())
        cpu.resize(m, n);
      
      cpu.fill_zero();
    }
    
    REAL_OUT *gbl = cpu.data_ptr();
    const REAL_IN *sub = mpi.data_ptr();
    
    cpumat<REAL_OUT> tmp(mb, nb);
    REAL_OUT *tmp_d = tmp.data_ptr();
    
    for (len_t gj=0; gj<n; gj+=nb)
    {
      const int pc = fml::bcutils::g2p(gj, nb, g.npcol());
      const len_t j = fml::bcutils::g2l(gj, nb, g.npcol());
      const len_t col_copylen = std::min(nb, n-gj);
      
      for (len_t gi=0; gi<m; gi+=mb)
      {
        const int pr = fml::bcutils::g2p(gi, mb, g.nprow());
        const len_t i = fml::bcutils::g2l(gi, mb, g.nprow());
        const len_t row_copylen = std::min(mb, m-gi);
        
        if (i_am_ret)
        {
          if (pr == g.myrow() && pc == g.mycol())
          {
            for (int jj=0; jj<col_copylen; jj++)
              arraytools::copy(row_copylen, sub + i+m_local*(j+jj), gbl + gi+m*(gj+jj));
          }
          else
            g.recv(row_copylen, col_copylen, m, gbl + gi+m*gj, pr, pc);
        }
        else if (pr == g.myrow() && pc == g.mycol())
        {
          for (len_t jj=0; jj<col_copylen; jj++)
          {
            for (len_t ii=0; ii<row_copylen; ii++)
              tmp_d[ii + mb*jj] = (REAL_OUT) sub[i+ii + m_local*(j+jj)];
          }
          
          g.send(row_copylen, col_copylen, mb, tmp_d, rdest, cdest);
        }
      }
    }
  }
  
  
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> mpi2cpu(const mpimat<REAL> &mpi)
  {
    cpumat<REAL> cpu;
    mpi2cpu(mpi, cpu);
    
    return cpu;
  }
  
  
  
  /**
    @brief Copy data from a CPU object to an MPI object.
    
    @details The CPU matrix should be replicated across all processes in the
    grid. This is just for testing purposes and should not be used in
    production.
    
    @param[in] cpu Input data.
    @param[out] mpi Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method has no communication.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2mpi(const cpumat<REAL_IN> &cpu, mpimat<REAL_OUT> &mpi)
  {
    const len_t m = cpu.nrows();
    const len_t n = cpu.ncols();
    
    if (m != mpi.nrows() || n != mpi.ncols())
      mpi.resize(m, n);
    
    mpi.fill_zero();
    
    const grid g = mpi.get_grid();
    
    const len_local_t m_local = mpi.nrows_local();
    const len_local_t n_local = mpi.ncols_local();
    const int mb = mpi.bf_rows();
    
    const REAL_IN *gbl = cpu.data_ptr();
    REAL_OUT *sub = mpi.data_ptr();
    
    if (m_local > 0 && n_local > 0)
    {
      for (len_local_t j=0; j<n_local; j++)
      {
        const int gj = fml::bcutils::l2g(j, mpi.bf_cols(), g.npcol(), g.mycol());
        
        for (len_local_t i=0; i<m_local; i+=mb)
        {
          const int gi = fml::bcutils::l2g(i, mpi.bf_rows(), g.nprow(), g.myrow());
          
          for (int ii=0; ii<mb && ii+i<m_local; ii++)
            sub[i+ii + m_local*j] = (REAL_OUT) gbl[gi+ii + m*gj];
        }
      }
    }
  }
  
  
  
  /**
    @brief Copy data from an MPI object to another.
    
    @param[in] mpi_in Input data.
    @param[out] mpi_out Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void mpi2mpi(const mpimat<REAL_IN> &mpi_in, mpimat<REAL_OUT> &mpi_out)
  {
    if (mpi_in.get_grid().ictxt() != mpi_out.get_grid().ictxt())
      throw std::runtime_error("mpimat objects must be distributed on the same process grid");
    
    mpi_out.resize(mpi_in.nrows(), mpi_in.ncols());
    
    size_t len = (size_t) mpi_in.nrows_local() * mpi_in.ncols_local();
    arraytools::copy(len, mpi_in.data_ptr(), mpi_out.data_ptr());
  }
}
}


#endif
