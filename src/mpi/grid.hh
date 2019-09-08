#ifndef FML_MPI_GRID_H
#define FML_MPI_GRID_H


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

#include "_blacs_prototypes.h"


/**
  @defgroup Enumerations
  Public enumeration types
*/

/**
 * @ingroup Enumerations
 * @brief Supported process grid shapes for 2-dimensional BLACS grids.
 *
 * These are the values used in grid constructor (parameter 'gridtype').
 */
enum gridshape
{
  /// A square process grid, or as square as can be when the total number of MPI
  /// ranks is not a perfect square. In the latter case, the grid size will be
  /// taken to be \f$ p_1 \times p_2 \f$ where \f$ p_1 > p_2 \f$ are integers of
  /// whose difference is as small as possible such that their product is the
  /// total number of MPI ranks. For example, with 210 processes we have
  /// \f$ 210=7*5*3*2=15*14 \f$, and with 14 we have \f$ 14=7*2 \f$.
  PROC_GRID_SQUARE,
  /// A grid with 1 row and as many columns as there are MPI ranks.
  PROC_GRID_WIDE,
  /// A grid with 1 column and as many rows as there are MPI ranks.
  PROC_GRID_TALL
};


/**
 * @brief 2-dimensional MPI process grid. 
 */
class grid
{
  public:
    grid();
    grid(int gridtype);
    
    void inherit_grid(int blacs_context);
    
    void exit();
    void finalize(bool mpi_continue=false);
    
    void printf(int row, int col, const char *fmt, ...) const;
    void info() const;
    
    bool rank0() const;
    void barrier(char scope) const;
    
    void send(int m, int n, int *x, int rdest, int cdest) const;
    void send(int m, int n, float *x, int rdest, int cdest) const;
    void send(int m, int n, double *x, int rdest, int cdest) const;
    
    void recv(int m, int n, int *x, int rdest, int cdest) const;
    void recv(int m, int n, float *x, int rdest, int cdest) const;
    void recv(int m, int n, double *x, int rdest, int cdest) const;
    
    void allreduce(int m, int n, int *x, char scope) const;
    void allreduce(int m, int n, float *x, char scope) const;
    void allreduce(int m, int n, double *x, char scope) const;
    
    void reduce(int m, int n, int *x, char scope, int rdest, int cdest) const;
    void reduce(int m, int n, float *x, char scope, int rdest, int cdest) const;
    void reduce(int m, int n, double *x, char scope, int rdest, int cdest) const;
    
    void bcast(int m, int n, int *x, char scope, int rsrc, int csrc) const;
    void bcast(int m, int n, float *x, char scope, int rsrc, int csrc) const;
    void bcast(int m, int n, double *x, char scope, int rsrc, int csrc) const;
    
    
    ///@{
    /// The BLACS integer context.
    int ictxt() const {return _ictxt;};
    /// The total number of processes bound to the BLACS context.
    int nprocs() const {return _nprocs;};
    /// The number of processes rows in the BLACS context.
    int nprow() const {return _nprow;};
    /// The number of processes columns in the BLACS context.
    int npcol() const {return _npcol;};
    /// The process row (0-based index) of the calling process.
    int myrow() const {return _myrow;};
    /// The process column (0-based index) of the calling process.
    int mycol() const {return _mycol;};
    ///@}
    
    /// Is the BLACS grid valid
    bool valid_grid() const {return (_ictxt!=UNINITIALIZED_GRID && _ictxt!=EXITED_GRID);};
  
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
    
    void squarish(int *nr, int *nc) const;
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
 * @brief Create a new grid object. Does not initialize any BLACS or MPI data.
*/
inline grid::grid()
{
  _ictxt = UNINITIALIZED_GRID;
  _nprocs = _nprow = _npcol = _myrow = _mycol = -1;
}



/**
 * @brief Create a new grid object by initializing a new BLACS process grid.
 * 
 * @param[in] gridtype Should be one of PROC_GRID_SQUARE, PROC_GRID_WIDE, or
 * PROC_GRID_TALL. Otherwise this will throw a 'runtime_error' exception.
*/
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



/**
 * @brief Create a grid object from an existing BLACS process grid.
 * 
 * @param blacs_context The BLACS integer context number.
*/
inline void grid::inherit_grid(int blacs_context)
{
  _ictxt = blacs_context;
  Cblacs_gridinfo(_ictxt, &_nprow, &_npcol, &_myrow, &_mycol);
  
  if (_nprow == -1)
    throw std::runtime_error("context handle does not point at a valid context");
  
  _nprocs = _nprow * _npcol;
}



// mpi/blacs cleanup

/**
 * @brief Exits the BLACS grid, but does not shutdown BLACS/MPI.
*/
inline void grid::exit()
{
  if (_ictxt != EXITED_GRID && _ictxt != UNINITIALIZED_GRID)
    Cblacs_gridexit(_ictxt);
  
  _ictxt = EXITED_GRID;
  _nprocs = _nprow = _npcol = _myrow = _mycol = -1;
}



/**
 * @brief Shuts down BLACS, and optionally MPI.
 * 
 * @param mpi_continue Should MPI continue, i.e., not be shut down too?
*/
inline void grid::finalize(bool mpi_continue)
{
  exit();
  
  int cont = (int) mpi_continue;
  Cblacs_exit(cont);
}



// printers

/**
 * @brief Helper wrapper around the C standard I/O 'printf()' function.
 * Conceptually similar to guarding a normal 'printf()' function with a check
 * for 'row==myrow() && col==mycol()'.
 * 
 * @param[in] row,col The process row/column that should do the printing.
 * @param[in] fmt The printf format string.
 * @param[in] ... additional arguments to printf.
*/
inline void grid::printf(int row, int col, const char *fmt, ...) const
{
  if (_myrow == row && _mycol == col)
  {
    va_list args;
    
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
  }
}



/**
 * @brief Print some brief information about the BLACS grid. The printing is
 * done by row 0 and col 0.
*/
inline void grid::info() const
{
  printf(0, 0, "## Grid %d %dx%d\n\n", _ictxt, _nprow, _npcol);
}



// misc

/**
 * @brief Check if the executing process is rank 0, i.e., if the process row and
 * column are 0.
*/
inline bool grid::rank0() const
{
  return (_myrow==0 && _mycol==0);
}



/**
 * @brief Execute a barrier across the specified scope of the BLACS grid.
 * 
 * @param scope The scope of the operation. For just rows use 'R', just columns
 use 'C', and for all processes use 'A'.
*/
inline void grid::barrier(char scope) const
{
  Cblacs_barrier(_ictxt, &scope);
}



/**
 * @brief Point-to-point send. Should be matched by a corresponding 'recv' call.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in] x The data to send.
 * @param[in] rdest,cdest The row/col destination in the BLACS grid.
*/
///@{
inline void grid::send(int m, int n, int *x, int rdest, int cdest) const
{
  Cigesd2d(_ictxt, m, n, x, m, rdest, cdest);
}

inline void grid::send(int m, int n, float *x, int rdest, int cdest) const
{
  Csgesd2d(_ictxt, m, n, x, m, rdest, cdest);
}

inline void grid::send(int m, int n, double *x, int rdest, int cdest) const
{
  Cdgesd2d(_ictxt, m, n, x, m, rdest, cdest);
}
///@}



/**
 * @brief Point-to-point receive. Should be matched by a corresponding 'send'
 call.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in] x The data to receive.
 * @param[in] rdest,cdest The row/col destination in the BLACS grid.
*/
///@{
inline void grid::recv(int m, int n, int *x, int rsrc, int csrc) const
{
  Cigerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}

inline void grid::recv(int m, int n, float *x, int rsrc, int csrc) const
{
  Csgerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}

inline void grid::recv(int m, int n, double *x, int rsrc, int csrc) const
{
  Cdgerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}
///@}



/**
 * @brief Sum reduce operation across all processes in the grid.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in,out] x The data to reduce.
 * @param scope The scope of the operation. For just rows use 'R', just columns
 use 'C', and for all processes use 'A'.
*/
///@{
inline void grid::allreduce(int m, int n, int *x, char scope) const
{
  char top = ' ';
  Cigsum2d(_ictxt, &scope, &top, m, n, x, m, -1, -1);
}

inline void grid::allreduce(int m, int n, float *x, char scope) const
{
  char top = ' ';
  Csgsum2d(_ictxt, &scope, &top, m, n, x, m, -1, -1);
}

inline void grid::allreduce(int m, int n, double *x, char scope) const
{
  char top = ' ';
  Cdgsum2d(_ictxt, &scope, &top, m, n, x, m, -1, -1);
}
///@}



/**
 * @brief Sum reduce operation.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in,out] x The data to reduce.
 * @param scope The scope of the operation. For just rows use 'R', just columns
 use 'C', and for all processes use 'A'.
 * @param[in] rdest,cdest The row/column of the BLACS grid to receive the final
 answer.
*/
///@{
inline void grid::reduce(int m, int n, int *x, char scope, int rdest, int cdest) const
{
  char top = ' ';
  Cigsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
}

inline void grid::reduce(int m, int n, float *x, char scope, int rdest, int cdest) const
{
  char top = ' ';
  Csgsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
}

inline void grid::reduce(int m, int n, double *x, char scope, int rdest, int cdest) const
{
  char top = ' ';
  Cdgsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
}
///@}



/**
 * @brief Broadcast.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in,out] x The data to reduce.
 * @param scope The scope of the operation. For just rows use 'R', just columns
 use 'C', and for all processes use 'A'.
 * @param[in] rsrc,csrc The process row/column of the BLACS grid that is
 broadcasting.
*/
///@{
inline void grid::bcast(int m, int n, int *x, char scope, int rsrc, int csrc) const
{
  char top = ' ';
  if (rsrc == _myrow && csrc == _mycol)
    Cigebs2d(_ictxt, &scope, &top, m, n, x, m);
  else
    Cigebr2d(_ictxt, &scope, &top, m, n, x, m, rsrc, csrc);
}

inline void grid::bcast(int m, int n, float *x, char scope, int rsrc, int csrc) const
{
  char top = ' ';
  if (rsrc == _myrow && csrc == _mycol)
    Csgebs2d(_ictxt, &scope, &top, m, n, x, m);
  else
    Csgebr2d(_ictxt, &scope, &top, m, n, x, m, rsrc, csrc);
}

inline void grid::bcast(int m, int n, double *x, char scope, int rsrc, int csrc) const
{
  char top = ' ';
  if (rsrc == _myrow && csrc == _mycol)
    Cdgebs2d(_ictxt, &scope, &top, m, n, x, m);
  else
    Cdgebr2d(_ictxt, &scope, &top, m, n, x, m, rsrc, csrc);
}
///@}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

inline void grid::squarish(int *nr, int *nc) const
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
