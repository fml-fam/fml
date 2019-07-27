#ifndef FML_MPI_GRID_H
#define FML_MPI_GRID_H


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>

#include "_blacs_prototypes.h"


enum gridshape {PROC_GRID_SQUARE, PROC_GRID_WIDE, PROC_GRID_TALL};


class grid
{
  public:
    grid();
    grid(int gridtype);
    grid(const grid &g);
    
    void exit();
    void finalize(int mpi_continue=0);
    void inherit_grid(int blacs_context);
    void printf(int row, int col, const char *fmt, ...);
    void print();
    
    void barrier(char *scope);
    
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



grid::grid()
{
  _ictxt = UNINITIALIZED_GRID;
  _nprocs = _nprow = _npcol = _myrow = _mycol = -1;
}



grid::grid(int gridtype)
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
  {
    // TODO
    // return FML_EXIT_ERROR_BLACSGRID;
  }
  
  Cblacs_gridinfo(_ictxt, &_nprow, &_npcol, &_myrow, &_mycol);
}



grid::grid(const grid &g)
{
  _nprocs = nprocs();
  _ictxt = ictxt();
  _nprow = nprow();
  _npcol = npcol();
  _myrow = myrow();
  _mycol = mycol();
}



void grid::exit()
{
  if (_ictxt != EXITED_GRID && _ictxt != UNINITIALIZED_GRID)
    Cblacs_gridexit(_ictxt);
  
  _ictxt = EXITED_GRID;
  _nprocs = _nprow = _npcol = _myrow = _mycol = -1;
}



void grid::finalize(int mpi_continue)
{
  exit();
  Cblacs_exit(mpi_continue);
}



void grid::inherit_grid(int blacs_context)
{
  // TODO size
  _ictxt = blacs_context;
  Cblacs_gridinfo(_ictxt, &_nprow, &_npcol, &_myrow, &_mycol);
}



void grid::printf(int row, int col, const char *fmt, ...)
{
  if (_myrow == row && _mycol == col)
  {
    va_list args;
    
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
  }
}



void grid::print()
{
  printf(0, 0, "## Grid %d %dx%d\n", _ictxt, _nprow, _npcol);
}



void grid::barrier(char *scope)
{
  Cblacs_barrier(_ictxt, scope);
}



void grid::squarish(int *nr, int *nc)
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
