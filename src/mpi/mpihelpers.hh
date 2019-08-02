#ifndef FML_MPIHELPERS_H
#define FML_MPIHELPERS_H


#include <stdexcept>

#include "../cpu/cpumat.hh"
#include "bcutils.hh"
#include "grid.hh"
#include "mpimat.hh"


namespace mpihelpers
{
  template <typename REAL>
  void mpi2cpu_noalloc(mpimat<REAL> &mpi, cpumat<REAL> &cpu)
  {
    len_t m = mpi.nrows();
    len_t n = mpi.ncols();
    
    if (m != cpu.nrows() || n != cpu.nrows())
      throw std::runtime_error("non-conformable arguments");
    
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
  cpumat<REAL> mpi2cpu(mpimat<REAL> &mpi)
  {
    len_t m = mpi.nrows();
    len_t n = mpi.ncols();
    
    cpumat<REAL> cpu(m, n);
    mpi2cpu_noalloc(mpi, cpu);
    
    return cpu;
  }
  
  
  
  template <typename REAL>
  mpimat<REAL> cpu2mpi(cpumat<REAL> &cpu, grid g, int bf_rows=16, int bf_cols=16)
  {
    len_t m = cpu.nrows();
    len_t n = cpu.ncols();
    
    mpimat<REAL> mpi(g, m, n, bf_rows, bf_cols);
    mpi.fill_zero();
    
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
    
    return mpi;
  }
}


#endif
