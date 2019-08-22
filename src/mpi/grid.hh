#ifndef FML_MPI_GRID_H
#define FML_MPI_GRID_H


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

#include "_blacs_prototypes.h"


enum gridshape {PROC_GRID_SQUARE, PROC_GRID_WIDE, PROC_GRID_TALL};


class grid
{
  public:
    grid();
    grid(int gridtype);
    
    void inherit_grid(int blacs_context);
    
    void exit();
    void finalize(int mpi_continue=0);
    
    void printf(int row, int col, const char *fmt, ...);
    void info();
    
    bool rank0();
    void barrier(char scope);
    
    void send(int m, int n, int *x, int rdest, int cdest);
    void send(int m, int n, float *x, int rdest, int cdest);
    void send(int m, int n, double *x, int rdest, int cdest);
    void recv(int m, int n, int *x, int rdest, int cdest);
    void recv(int m, int n, float *x, int rdest, int cdest);
    void recv(int m, int n, double *x, int rdest, int cdest);
    
    void reduce(int m, int n, int *x, char scope, int rdest, int cdest);
    void reduce(int m, int n, float *x, char scope, int rdest, int cdest);
    void reduce(int m, int n, double *x, char scope, int rdest, int cdest);
    
    int ictxt() const {return _ictxt;};
    int nprocs() const {return _nprocs;};
    int nprow() const {return _nprow;};
    int npcol() const {return _npcol;};
    int myrow() const {return _myrow;};
    int mycol() const {return _mycol;};
  
  protected:
    int _ictxt;
    int _nprocs;
    int _nprow;
    int _npcol;
    int _myrow;
    int _mycol;
  
  private:
    static const int UNINITIALIZED_GRID = -1;
    static const int EXITED_GRID = -11;
    
    void squarish(int *nr, int *nc);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

inline grid::grid()
{
  _ictxt = UNINITIALIZED_GRID;
  _nprocs = _nprow = _npcol = _myrow = _mycol = -1;
}



inline grid::grid(int gridtype)
{
  char order = 'R';
  
  int mypnum;
  Cblacs_pinfo(&mypnum, &_nprocs);
  
  Cblacs_get(-1, 0, &_ictxt);
  
  if (gridtype == PROC_GRID_SQUARE)
  {
    int nr, nc;
    squarish(&nr, &nc);
    
    Cblacs_gridinit(&_ictxt, &order, nr, nc);
  }
  else if (gridtype == PROC_GRID_TALL)
    Cblacs_gridinit(&_ictxt, &order, _nprocs, 1);
  else if (gridtype == PROC_GRID_WIDE)
    Cblacs_gridinit(&_ictxt, &order, 1, _nprocs);
  else
    throw std::runtime_error("Process grid should be one of PROC_GRID_SQUARE, PROC_GRID_TALL, or PROC_GRID_WIDE");
  
  Cblacs_gridinfo(_ictxt, &_nprow, &_npcol, &_myrow, &_mycol);
}



inline void grid::inherit_grid(int blacs_context)
{
  _ictxt = blacs_context;
  Cblacs_gridinfo(_ictxt, &_nprow, &_npcol, &_myrow, &_mycol);
  _nprocs = _nprow * _npcol;
}



// mpi/blacs cleanup

inline void grid::exit()
{
  if (_ictxt != EXITED_GRID && _ictxt != UNINITIALIZED_GRID)
    Cblacs_gridexit(_ictxt);
  
  _ictxt = EXITED_GRID;
  _nprocs = _nprow = _npcol = _myrow = _mycol = -1;
}



inline void grid::finalize(int mpi_continue)
{
  exit();
  Cblacs_exit(mpi_continue);
}



// printers

inline void grid::printf(int row, int col, const char *fmt, ...)
{
  if (_myrow == row && _mycol == col)
  {
    va_list args;
    
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
  }
}



inline void grid::info()
{
  printf(0, 0, "## Grid %d %dx%d\n\n", _ictxt, _nprow, _npcol);
}



// misc

inline bool grid::rank0()
{
  return (_myrow==0 && _mycol==0);
}



inline void grid::barrier(char scope)
{
  Cblacs_barrier(_ictxt, &scope);
}



// send/recv

inline void grid::send(int m, int n, int *x, int rdest, int cdest)
{
  Cigesd2d(_ictxt, m, n, x, m, rdest, cdest);
}

inline void grid::send(int m, int n, float *x, int rdest, int cdest)
{
  Csgesd2d(_ictxt, m, n, x, m, rdest, cdest);
}

inline void grid::send(int m, int n, double *x, int rdest, int cdest)
{
  Cdgesd2d(_ictxt, m, n, x, m, rdest, cdest);
}



inline void grid::recv(int m, int n, int *x, int rsrc, int csrc)
{
  Cigerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}

inline void grid::recv(int m, int n, float *x, int rsrc, int csrc)
{
  Csgerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}

inline void grid::recv(int m, int n, double *x, int rsrc, int csrc)
{
  Cdgerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}



// reductions

inline void grid::reduce(int m, int n, int *x, char scope, int rdest, int cdest)
{
  char top = ' ';
  Cigsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
}

inline void grid::reduce(int m, int n, float *x, char scope, int rdest, int cdest)
{
  char top = ' ';
  Csgsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
}

inline void grid::reduce(int m, int n, double *x, char scope, int rdest, int cdest)
{
  char top = ' ';
  Cdgsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

inline void grid::squarish(int *nr, int *nc)
{
  int n = (int) sqrt((double) _nprocs);
  n = (n<1)?1:n; // suppresses bogus compiler warning
  
  for (int i=0; i<n; i++)
  {
    (*nc) = n - i;
    (*nr) = _nprocs % (*nc);
    if ((*nr) == 0)
      break;
  }
  
  (*nr) = _nprocs / (*nc);
}


#endif
