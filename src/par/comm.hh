#ifndef FML_PAR_COMM_H
#define FML_PAR_COMM_H


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "../types.hh"


/**
 * @brief MPI communicator data and helpers. 
 */
class comm
{
  public:
    comm();
    
    void inherit_comm(MPI_Comm comm);
    
    void finalize();
    
    void printf(int rank, const char *fmt, ...);
    void info();
    
    bool rank0();
    std::vector<int> jid(const int n);
    void barrier();
    
    void send(int n, int *data, int dest, int tag=0);
    void send(int n, float *data, int dest, int tag=0);
    void send(int n, double *data, int dest, int tag=0);
    
    void recv(int n, int *data, int source, int tag=0);
    void recv(int n, float *data, int source, int tag=0);
    void recv(int n, double *data, int source, int tag=0);
    
    void allreduce(int n, int *data);
    void allreduce(int n, len_global_t *data);
    void allreduce(int n, float *data);
    void allreduce(int n, double *data);
    
    void reduce(int n, int *data, int root=0);
    void reduce(int n, len_global_t *data, int root=0);
    void reduce(int n, float *data, int root=0);
    void reduce(int n, double *data, int root=0);
    
    void bcast(int n, int *data, int root);
    void bcast(int n, float *data, int root);
    void bcast(int n, double *data, int root);
    
    ///@{
    /// The MPI communicator.
    MPI_Comm get_comm() const {return _comm;};
    /// Calling process rank (0-based index) in the MPI communicator.
    int rank() const {return _rank;};
    /// Total number of ranks in the MPI communicator.
    int size() const {return _size;};
    ///@}
  
  protected:
    MPI_Comm _comm;
    int _rank;
    int _size;
  
  private:
    void init();
    void set_metadata();
    void check_ret(int ret);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
 * @brief Create a new comm object and uses 'MPI_COMM_WORLD' as the
   communicator.
 */
inline comm::comm()
{
  init();
  
  _comm = MPI_COMM_WORLD;
  
  set_metadata();
}



/**
 * @brief Change communicator to an existing one.
 * 
 * @param comm An MPI communicator.
 */
inline void comm::inherit_comm(MPI_Comm comm)
{
  _comm = comm;
  set_metadata();
}



/**
 * @brief Shut down MPI.
 */
inline void comm::finalize()
{
  int ret = MPI_Finalize();
  check_ret(ret);
}



// printers

/**
 * @brief Helper wrapper around the C standard I/O 'printf()' function.
 * Conceptually similar to guarding a normal 'printf()' function with a check
 * for 'rank==rank()'.
 * 
 * @param[in] rank The process that should do the printing.
 * @param[in] fmt The printf format string.
 * @param[in] ... additional arguments to printf.
 */
inline void comm::printf(int rank, const char *fmt, ...)
{
  if (_rank == rank)
  {
    va_list args;
    
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
  }
}



/**
 * @brief Print some brief information about the MPI communicator. The printing
   is done by rank 0.
 */
inline void comm::info()
{
  printf(0, "## MPI on %d ranks\n\n", _size);
}



// misc

/**
 * @brief Check if the executing process is rank 0.
 */
inline bool comm::rank0()
{
  return (_rank == 0);
}



/**
 * @brief 
 * 
 * @param[in] n The number of tasks.
 * @return A std::vector of task numbers.
 * 
 * @comm The method has no communication.
 */
inline std::vector<int> comm::jid(const int n)
{
  std::vector<int> ret;
  
  if (n > _size)
  {
    int local = n / _size;
    int rem = n % _size;
    
    if (rem == 0 || (_rank < (_size - rem)))
    {
      ret.resize(local);
      for (int i=0; i<local; i++)
        ret[i] = i + (_rank*local);
    }
    else
    {
      ret.resize(local+1);
      for (int i=0; i<=local; i++)
        ret[i] = i + (_rank*(local+1)) - (_size - rem);
    }
  }
  else
  {
    if (n > _rank)
    {
      ret.resize(1);
      ret[0] = _rank;
    }
    else
      ret.resize(0);
  }
  
  return ret;
}



/**
 * @brief Execute a barrier.
 */
inline void comm::barrier()
{
  int ret = MPI_Barrier(_comm);
  check_ret(ret);
}



// send/recv

/**
 * @brief Point-to-point send. Should be matched by a corresponding 'recv' call.
 * 
 * @param[in] n Number of elements of 'data'.
 * @param[in] data The data to send.
 * @param[in] dest The process destination in the MPI communicator.
 * @param[in] tag Optional MPI tag (default=0).
 */
///@{
inline void comm::send(int n, int *data, int dest, int tag)
{
  int ret = MPI_Send(data, n, MPI_INT, dest, tag, _comm);
  check_ret(ret);
}

inline void comm::send(int n, float *data, int dest, int tag)
{
  int ret = MPI_Send(data, n, MPI_FLOAT, dest, tag, _comm);
  check_ret(ret);
}

inline void comm::send(int n, double *data, int dest, int tag)
{
  int ret = MPI_Send(data, n, MPI_DOUBLE, dest, tag, _comm);
  check_ret(ret);
}
///@}



/**
 * @brief Point-to-point receive. Should be matched by a corresponding 'send'
   call.
 * 
 * @param[in] n Number of elements of 'data'.
 * @param[in] data The data to send.
 * @param[in] source The process source in the MPI communicator.
 * @param[in] tag Optional MPI tag (default=0).
 */
///@{
inline void comm::recv(int n, int *data, int source, int tag)
{
  int ret = MPI_Recv(data, n, MPI_INT, source, tag, _comm, MPI_STATUS_IGNORE);
  check_ret(ret);
}

inline void comm::recv(int n, float *data, int source, int tag)
{
  int ret = MPI_Recv(data, n, MPI_FLOAT, source, tag, _comm, MPI_STATUS_IGNORE);
  check_ret(ret);
}

inline void comm::recv(int n, double *data, int source, int tag)
{
  int ret = MPI_Recv(data, n, MPI_DOUBLE, source, tag, _comm, MPI_STATUS_IGNORE);
  check_ret(ret);
}
///@}



// reductions

/**
 * @brief Sum reduce operation across all processes in the MPI communicator.
 * 
 * @param[in] n Number of elemends of 'data'.
 * @param[in,out] data The data to reduce.
 */
///@{
inline void comm::allreduce(int n, int *data)
{
  int ret = MPI_Allreduce(MPI_IN_PLACE, data, n, MPI_INT, MPI_SUM, _comm);
  check_ret(ret);
}

inline void comm::allreduce(int n, len_global_t *data)
{
  int ret = MPI_Allreduce(MPI_IN_PLACE, data, n, MPI_LENGLOBAL_T, MPI_SUM, _comm);
  check_ret(ret);
}

inline void comm::allreduce(int n, float *data)
{
  int ret = MPI_Allreduce(MPI_IN_PLACE, data, n, MPI_FLOAT, MPI_SUM, _comm);
  check_ret(ret);
}

inline void comm::allreduce(int n, double *data)
{
  int ret = MPI_Allreduce(MPI_IN_PLACE, data, n, MPI_DOUBLE, MPI_SUM, _comm);
  check_ret(ret);
}
///@}



/**
 * @brief Sum reduce operation across all processes in the MPI communicator.
 * 
 * @param[in] n Number of elemends of 'data'.
 * @param[in,out] data The data to reduce.
 * @param[in] root The rank in the MPI communicator to receive the final answer.
 */
///@{
inline void comm::reduce(int n, int *data, int root)
{
  int ret = MPI_Reduce(MPI_IN_PLACE, data, n, MPI_INT, MPI_SUM, root, _comm);
  check_ret(ret);
}

inline void comm::reduce(int n, len_global_t *data, int root)
{
  int ret = MPI_Reduce(MPI_IN_PLACE, data, n, MPI_LENGLOBAL_T, MPI_SUM, root, _comm);
  check_ret(ret);
}

inline void comm::reduce(int n, float *data, int root)
{
  int ret = MPI_Reduce(MPI_IN_PLACE, data, n, MPI_FLOAT, MPI_SUM, root, _comm);
  check_ret(ret);
}

inline void comm::reduce(int n, double *data, int root)
{
  int ret = MPI_Reduce(MPI_IN_PLACE, data, n, MPI_DOUBLE, MPI_SUM, root, _comm);
  check_ret(ret);
}
///@}



// broadcasters

/**
 * @brief Broadcast.
 * 
 * @param[in] n Number of elemends of 'data'.
 * @param[in,out] data The data to broadcast.
 * @param[in] root The rank in the MPI communicator that does the broadcasting.
 */
///@{
inline void comm::bcast(int n, int *data, int root)
{
  int ret = MPI_Bcast(data, n, MPI_INT, root, _comm);
  check_ret(ret);
}

inline void comm::bcast(int n, float *data, int root)
{
  int ret = MPI_Bcast(data, n, MPI_FLOAT, root, _comm);
  check_ret(ret);
}

inline void comm::bcast(int n, double *data, int root)
{
  int ret = MPI_Bcast(data, n, MPI_DOUBLE, root, _comm);
  check_ret(ret);
}
///@}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

inline void comm::init()
{
  int ret;
  int flag;
  
  ret = MPI_Initialized(&flag);
  check_ret(ret);
  
  if (!flag)
  {
    ret = MPI_Init(NULL, NULL);
    check_ret(ret);
  }
}



inline void comm::set_metadata()
{
  int ret;
  
  ret = MPI_Comm_rank(_comm, &_rank);
  check_ret(ret);
  
  ret = MPI_Comm_size(_comm, &_size);
  check_ret(ret);
}



inline void comm::check_ret(int ret)
{
  if (ret != MPI_SUCCESS && _rank == 0)
  {
    int slen;
    char s[MPI_MAX_ERROR_STRING];
    
    MPI_Error_string(ret, s, &slen);
    throw std::runtime_error(s);
  }
}


#endif
