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
 * @ingroup Enumerations
 * @brief Supported operations in reduce/allreduce.
 */
enum blacsops
{
  BLACS_SUM,
  BLACS_MAX,
  BLACS_MIN
};


/**
 * @brief 2-dimensional MPI process grid. 
 */
class grid
{
  public:
    // constructors/destructor and comm management
    grid();
    grid(const gridshape gridtype);
    void set(const int blacs_context);
    void exit();
    void finalize(const bool mpi_continue=false);
    
    // utilities
    void printf(const int row, const int col, const char *fmt, ...) const;
    void info() const;
    bool rank0() const;
    
    // send/recv
    void send(const int m, const int n, const int *x, const int rdest=0, const int cdest=0) const;
    void send(const int m, const int n, const float *x, const int rdest=0, const int cdest=0) const;
    void send(const int m, const int n, const double *x, const int rdest=0, const int cdest=0) const;
    
    void recv(const int m, const int n, int *x, const int rsrc=0, const int csrc=0) const;
    void recv(const int m, const int n, float *x, const int rsrc=0, const int csrc=0) const;
    void recv(const int m, const int n, double *x, const int rsrc=0, const int csrc=0) const;
    
    // collectives
    void barrier(const char scope) const;
    
    void allreduce(const int m, const int n, int *x, const char scope='A', const blacsops op=BLACS_SUM) const;
    void allreduce(const int m, const int n, float *x, const char scope='A', const blacsops op=BLACS_SUM) const;
    void allreduce(const int m, const int n, double *x, const char scope='A', const blacsops op=BLACS_SUM) const;
    
    void reduce(const int m, const int n, int *x, const char scope='A', const blacsops op=BLACS_SUM, const int rdest=0, const int cdest=0) const;
    void reduce(const int m, const int n, float *x, const char scope='A', const blacsops op=BLACS_SUM, const int rdest=0, const int cdest=0) const;
    void reduce(const int m, const int n, double *x, const char scope='A', const blacsops op=BLACS_SUM, const int rdest=0, const int cdest=0) const;
    
    void bcast(const int m, const int n, int *x, const char scope='A', const int rsrc=0, const int csrc=0) const;
    void bcast(const int m, const int n, float *x, const char scope='A', const int rsrc=0, const int csrc=0) const;
    void bcast(const int m, const int n, double *x, const char scope='A', const int rsrc=0, const int csrc=0) const;
    
    
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
    
    /// Is the BLACS grid valid?
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

// constructors/destructor and grid management

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
   PROC_GRID_TALL.
 * 
 * @except If 'gridtype' is not one of PROC_GRID_SQUARE, PROC_GRID_WIDE, or
   PROC_GRID_TALL, the method will throw a 'runtime_error' exception.
 */
inline grid::grid(const gridshape gridtype)
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
inline void grid::set(const int blacs_context)
{
  _ictxt = blacs_context;
  Cblacs_gridinfo(_ictxt, &_nprow, &_npcol, &_myrow, &_mycol);
  
  if (_nprow == -1)
    throw std::runtime_error("context handle does not point at a valid context");
  
  _nprocs = _nprow * _npcol;
}



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
inline void grid::finalize(const bool mpi_continue)
{
  exit();
  
  int cont = (int) mpi_continue;
  Cblacs_exit(cont);
}



// utilities

/**
 * @brief Helper wrapper around the C standard I/O 'printf()' function.
   Conceptually similar to guarding a normal 'printf()' function with a check
   for 'row==myrow() && col==mycol()'.
 * 
 * @param[in] row,col The process row/column that should do the printing.
 * @param[in] fmt The printf format string.
 * @param[in] ... additional arguments to printf.
 */
inline void grid::printf(const int row, const int col, const char *fmt, ...) const
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
   done by row 0 and col 0.
 */
inline void grid::info() const
{
  printf(0, 0, "## Grid %d %dx%d\n\n", _ictxt, _nprow, _npcol);
}



/**
 * @brief Check if the executing process is rank 0, i.e., if the process row and
   column are 0.
 */
inline bool grid::rank0() const
{
  return (_myrow==0 && _mycol==0);
}



// send/recv

/**
 * @brief Point-to-point send. Should be matched by a corresponding 'recv' call.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in] x The data to send.
 * @param[in] rdest,cdest The row/col destination in the BLACS grid.
 */
///@{
inline void grid::send(const int m, const int n, const int *x, const int rdest, const int cdest) const
{
  Cigesd2d(_ictxt, m, n, x, m, rdest, cdest);
}

inline void grid::send(const int m, const int n, const float *x, const int rdest, const int cdest) const
{
  Csgesd2d(_ictxt, m, n, x, m, rdest, cdest);
}

inline void grid::send(const int m, const int n, const double *x, const int rdest, const int cdest) const
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
 * @param[in] rsrc,csrc The row/col source in the BLACS grid.
 */
///@{
inline void grid::recv(const int m, const int n, int *x, const int rsrc, const int csrc) const
{
  Cigerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}

inline void grid::recv(const int m, const int n, float *x, const int rsrc, const int csrc) const
{
  Csgerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}

inline void grid::recv(const int m, const int n, double *x, const int rsrc, const int csrc) const
{
  Cdgerv2d(_ictxt, m, n, x, m, rsrc, csrc);
}
///@}



// collectives

/**
 * @brief Execute a barrier across the specified scope of the BLACS grid.
 * 
 * @param scope The scope of the operation. For just rows use 'R', just columns
   use 'C', and for all processes use 'A'.
 */
inline void grid::barrier(const char scope) const
{
  Cblacs_barrier(_ictxt, &scope);
}



/**
 * @brief Sum reduce operation across all processes in the grid.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in,out] x The data to reduce.
 * @param scope The scope of the operation. For just rows use 'R', just columns
   use 'C', and for all processes use 'A'.
 */
///@{
inline void grid::allreduce(const int m, const int n, int *x, const char scope, const blacsops op) const
{
  reduce(m, n, x, scope, op, -1, -1);
}

inline void grid::allreduce(const int m, const int n, float *x, const char scope, const blacsops op) const
{
  reduce(m, n, x, scope, op, -1, -1);
}

inline void grid::allreduce(const int m, const int n, double *x, const char scope, const blacsops op) const
{
  reduce(m, n, x, scope, op, -1, -1);
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
inline void grid::reduce(const int m, const int n, int *x, const char scope, const blacsops op, const int rdest, const int cdest) const
{
  char top = ' ';
  
  if (op == BLACS_SUM)
    Cigsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
  else if (op == BLACS_MAX)
    Cigamx2d(_ictxt, &scope, &top, m, n, x, m, NULL, NULL, -1, rdest, cdest);
  else if (op == BLACS_MIN)
    Cigamn2d(_ictxt, &scope, &top, m, n, x, m, NULL, NULL, -1, rdest, cdest);
}

inline void grid::reduce(const int m, const int n, float *x, const char scope, const blacsops op, const int rdest, const int cdest) const
{
  char top = ' ';
  
  if (op == BLACS_SUM)
    Csgsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
  else if (op == BLACS_MAX)
    Csgamx2d(_ictxt, &scope, &top, m, n, x, m, NULL, NULL, -1, rdest, cdest);
  else if (op == BLACS_MIN)
    Csgamn2d(_ictxt, &scope, &top, m, n, x, m, NULL, NULL, -1, rdest, cdest);
}

inline void grid::reduce(const int m, const int n, double *x, const char scope, const blacsops op, const int rdest, const int cdest) const
{
  char top = ' ';
  
  if (op == BLACS_SUM)
    Cdgsum2d(_ictxt, &scope, &top, m, n, x, m, rdest, cdest);
  else if (op == BLACS_MAX)
    Cdgamx2d(_ictxt, &scope, &top, m, n, x, m, NULL, NULL, -1, rdest, cdest);
  else if (op == BLACS_MIN)
    Cdgamn2d(_ictxt, &scope, &top, m, n, x, m, NULL, NULL, -1, rdest, cdest);
}
///@}



/**
 * @brief Broadcast.
 * 
 * @param[in] m,n Dimensions (number of rows/cols) of the data 'x'.
 * @param[in,out] x The data to broadcast.
 * @param scope The scope of the operation. For just rows use 'R', just columns
   use 'C', and for all processes use 'A'.
 * @param[in] rsrc,csrc The process row/column of the BLACS grid that is
   broadcasting.
 */
///@{
inline void grid::bcast(const int m, const int n, int *x, const char scope, const int rsrc, const int csrc) const
{
  char top = ' ';
  if (rsrc == _myrow && csrc == _mycol)
    Cigebs2d(_ictxt, &scope, &top, m, n, x, m);
  else
    Cigebr2d(_ictxt, &scope, &top, m, n, x, m, rsrc, csrc);
}

inline void grid::bcast(const int m, const int n, float *x, const char scope, const int rsrc, const int csrc) const
{
  char top = ' ';
  if (rsrc == _myrow && csrc == _mycol)
    Csgebs2d(_ictxt, &scope, &top, m, n, x, m);
  else
    Csgebr2d(_ictxt, &scope, &top, m, n, x, m, rsrc, csrc);
}

inline void grid::bcast(const int m, const int n, double *x, const char scope, const int rsrc, const int csrc) const
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
