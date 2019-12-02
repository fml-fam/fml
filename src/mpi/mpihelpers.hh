#ifndef FML_MPI_MPIHELPERS_H
#define FML_MPI_MPIHELPERS_H
#pragma once


#include <stdexcept>

#include "../cpu/cpumat.hh"
#include "bcutils.hh"
#include "grid.hh"
#include "mpimat.hh"


namespace mpihelpers
{
  template <typename REAL>
  void mpi2cpu(const mpimat<REAL> &mpi, cpumat<REAL> &cpu)
  {
    len_t m = mpi.nrows();
    len_t n = mpi.ncols();
    
    if (m != cpu.nrows() || n != cpu.ncols())
      cpu.resize(m, n);
    
    len_local_t m_local = mpi.nrows_local();
    len_local_t n_local = mpi.ncols_local();
    
    grid g = mpi.get_grid();
    
    cpu.fill_zero();
    
    REAL *gbl = cpu.data_ptr();
    REAL *sub = mpi.data_ptr();
    
    if (m_local > 0 && n_local > 0)
    {
      for (len_local_t j=0; j<n_local; j++)
      {
        for (len_local_t i=0; i<m_local; i++)
        {
          int gi = bcutils::l2g(i, mpi.bf_rows(), g.nprow(), g.myrow());
          int gj = bcutils::l2g(j, mpi.bf_cols(), g.npcol(), g.mycol());
          
          gbl[gi + m*gj] = sub[i + m_local*j];
        }
      }
    }
    
    g.reduce(m, n, gbl, 'A', 0, 0);
  }
  
  
  
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
    
    REAL *gbl = cpu.data_ptr();
    REAL *sub = mpi.data_ptr();
    
    if (m_local > 0 && n_local > 0)
    {
      for (len_local_t j=0; j<n_local; j++)
      {
        for (len_local_t i=0; i<m_local; i++)
        {
          int gi = bcutils::l2g(i, mpi.bf_rows(), g.nprow(), g.myrow());
          int gj = bcutils::l2g(j, mpi.bf_cols(), g.npcol(), g.mycol());
          
          sub[i + m_local*j] = gbl[gi + m*gj];
        }
      }
    }
  }
}


#endif
