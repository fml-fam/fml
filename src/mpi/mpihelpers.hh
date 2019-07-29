#ifndef FML_MPIHELPERS_H
#define FML_MPIHELPERS_H


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
        for (len_local_t i=0; i<n_local; i++)
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
}


#endif
