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
    
    len_local_t m_local = mpi.nrows_local();
    len_local_t n_local = mpi.ncols_local();
    
    int mb = mpi.bf_rows();
    
    len_t m = mpi.nrows();
    len_t n = mpi.ncols();
    
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
    grid g = mpi.get_grid();
    if (!g.ingrid())
      return;
    
    bool i_am_ret = (g.myrow() == rdest && g.mycol() == cdest) ? true : false;
    
    len_local_t m_local = mpi.nrows_local();
    
    int mb = mpi.bf_rows();
    int nb = mpi.bf_cols();
    
    len_t m = mpi.nrows();
    len_t n = mpi.ncols();
    
    if (i_am_ret)
    {
      if (m != cpu.nrows() || n != cpu.ncols())
        cpu.resize(m, n);
      
      cpu.fill_zero();
    }
    
    REAL_OUT *gbl = cpu.data_ptr();
    const REAL_IN *sub = mpi.data_ptr();
    
    for (len_t gj=0; gj<n; gj+=nb)
    {
      const int pc = fml::bcutils::g2p(gj, nb, g.npcol());
      const int j = fml::bcutils::g2l(gj, nb, g.npcol());
      
      for (len_t gi=0; gi<m; gi+=mb)
      {
        const int pr = fml::bcutils::g2p(gi, mb, g.nprow());
        const int i = fml::bcutils::g2l(gi, mb, g.nprow());
        
        const int row_copylen = std::min(mb, m-gi);
        const int col_copylen = std::min(nb, n-gj);
        
        if (i_am_ret)
        {
          if (pr == g.myrow() && pc == g.mycol())
          {
            for (int jj=0; jj<col_copylen; jj++)
              arraytools::copy(row_copylen, sub + i+m_local*(j+jj), gbl + gi+m*(gj+jj));
          }
          else
            g.recv(row_copylen, col_copylen, m, gbl + gi+m*j, pr, pc);
        }
        else if (pr == g.myrow() && pc == g.mycol())
          g.send(row_copylen, col_copylen, m_local, sub + i+m_local*j, rdest, cdest);
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
    
    @tparam REAL Should be `float` or `double`.
  */
  template <typename REAL>
  void cpu2mpi(const cpumat<REAL> &cpu, mpimat<REAL> &mpi)
  {
    len_t m = cpu.nrows();
    len_t n = cpu.ncols();
    
    if (m != mpi.nrows() || n != mpi.ncols())
      mpi.resize(m, n);
    
    mpi.fill_zero();
    
    grid g = mpi.get_grid();
    
    len_local_t m_local = mpi.nrows_local();
    len_local_t n_local = mpi.ncols_local();
    
    const REAL *gbl = cpu.data_ptr();
    REAL *sub = mpi.data_ptr();
    
    if (m_local > 0 && n_local > 0)
    {
      for (len_local_t j=0; j<n_local; j++)
      {
        const int gj = fml::bcutils::l2g(j, mpi.bf_cols(), g.npcol(), g.mycol());
        
        #pragma omp for simd
        for (len_local_t i=0; i<m_local; i++)
        {
          const int gi = fml::bcutils::l2g(i, mpi.bf_rows(), g.nprow(), g.myrow());
          
          sub[i + m_local*j] = gbl[gi + m*gj];
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


#endif
