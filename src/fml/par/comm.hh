// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_COMM_H
#define FML_PAR_COMM_H
#pragma once


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <typeinfo>
#include <vector>


namespace fml
{
  /// @brief MPI communicator data and helpers. 
  class comm
  {
    public:
      // constructors/destructor and comm management
      comm(MPI_Comm comm=MPI_COMM_WORLD);
      void set(MPI_Comm comm);
      comm create(MPI_Group group);
      comm split(int color, int key);
      void free();
      void finalize();
      
      // utilities
      void printf(int rank, const char *fmt, ...) const;
      void info() const;
      bool rank0() const;
      std::vector<int> jid(const int n) const;
      
      // send/recv
      template <typename T>
      void send(int n, const T *data, int dest, int tag=0) const;
      template <typename T>
      void isend(int n, const T *data, int dest, int tag=0) const;
      template <typename T>
      void recv(int n, T *data, int source, int tag=0) const;
      template <typename T>
      void irecv(int n, T *data, int source, int tag=0) const;
      
      // collectives
      void barrier() const;
      template <typename T>
      void allreduce(int n, T *data, MPI_Op op=MPI_SUM) const;
      template <typename T>
      void reduce(int n, T *data, MPI_Op op=MPI_SUM, int root=0) const;
      template <typename T>
      void bcast(int n, T *data, int root) const;
      
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
      void check_ret(const int ret) const;
      template <typename T>
      MPI_Datatype mpi_type_lookup(const T *x) const;
  };
}



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor and comm management

/**
  @brief Create a new comm object and uses 'MPI_COMM_WORLD' as the
   communicator.
  
  @param[in] comm An MPI communicator. Default is MPI_COMM_WORLD.
 */
inline fml::comm::comm(MPI_Comm comm)
{
  init();
  
  _comm = comm;
  
  set_metadata();
}



/**
  @brief Change communicator to an existing one.
  
  @param[in] comm An MPI communicator.
 */
inline void fml::comm::set(MPI_Comm comm)
{
  _comm = comm;
  set_metadata();
}



/**
  @brief Create new communicator based on color/key.
  
  @param[in] group 
  @return The new communicator managed in a new comm object.
 */
inline fml::comm fml::comm::create(MPI_Group group)
{
  MPI_Comm newcomm;
  int mpi_ret = MPI_Comm_create(_comm, group, &newcomm);
  check_ret(mpi_ret);
  
  fml::comm ret(newcomm);
  return ret;
}



/**
  @brief Create new communicator based on color/key.
  
  @param[in] color 
  @param[in] key 
  @return The new communicator managed in a new comm object.
 */
inline fml::comm fml::comm::split(int color, int key)
{
  MPI_Comm newcomm;
  int mpi_ret = MPI_Comm_split(_comm, color, key, &newcomm);
  check_ret(mpi_ret);
  
  fml::comm ret(newcomm);
  return ret;
}



/// @brief Destroy communicator.
inline void fml::comm::free()
{
  int mpi_ret = MPI_Comm_free(&_comm);
  check_ret(mpi_ret);
  
  _comm = MPI_COMM_NULL;
  _rank = -1;
  _size = -1;
}



/// @brief Shut down MPI.
inline void fml::comm::finalize()
{
  int ret = MPI_Finalize();
  check_ret(ret);
}



// utilities

/**
  @brief Helper wrapper around the C standard I/O 'printf()' function.
  Conceptually similar to guarding a normal 'printf()' function with a check
  for 'rank==rank()'.
  
  @param[in] rank The process that should do the printing.
  @param[in] fmt The printf format string.
  @param[in] ... additional arguments to printf.
 */
inline void fml::comm::printf(int rank, const char *fmt, ...) const
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
  @brief Print some brief information about the MPI communicator. The printing
  is done by rank 0.
 */
inline void fml::comm::info() const
{
  printf(0, "## MPI on %d ranks\n\n", _size);
}



/// @brief Check if the executing process is rank 0.
inline bool fml::comm::rank0() const
{
  return (_rank == 0);
}



/**
  @brief 
  
  @param[in] n The number of tasks.
  @return A std::vector of task numbers.
  
  @comm The method has no communication.
 */
inline std::vector<int> fml::comm::jid(const int n) const
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



// send/recv

/**
  @brief Point-to-point send. Should be matched by a corresponding 'recv' call.
  
  @param[in] n Number of elements of 'data'.
  @param[in] data The data to send.
  @param[in] dest The process destination in the MPI communicator.
  @param[in] tag Optional MPI tag (default=0).
 */
///@{
template <typename T>
inline void fml::comm::send(int n, const T *data, int dest, int tag) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Send(data, n, type, dest, tag, _comm);
  check_ret(ret);
}

template <typename T>
inline void fml::comm::isend(int n, const T *data, int dest, int tag) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Isend(data, n, type, dest, tag, _comm);
  check_ret(ret);
}
///@}



/**
  @brief Point-to-point receive. Should be matched by a corresponding 'send'
  call.
  
  @param[in] n Number of elements of 'data'.
  @param[in] data The data to send.
  @param[in] source The process source in the MPI communicator.
  @param[in] tag Optional MPI tag (default=0).
 */
///@{
template <typename T>
inline void fml::comm::recv(int n, T *data, int source, int tag) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Recv(data, n, type, source, tag, _comm, MPI_STATUS_IGNORE);
  check_ret(ret);
}

template <typename T>
inline void fml::comm::irecv(int n, T *data, int source, int tag) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Irecv(data, n, type, source, tag, _comm, MPI_STATUS_IGNORE);
  check_ret(ret);
}
///@}



// collectives

/**
  @brief Execute a barrier.
 */
inline void fml::comm::barrier() const
{
  int ret = MPI_Barrier(_comm);
  check_ret(ret);
}



/**
  @brief Sum reduce operation across all processes in the MPI communicator.
  
  @param[in] n Number of elemends of 'data'.
  @param[in,out] data The data to reduce.
  @param[in] op MPI reduction operation. Default is MPI_SUM.
 */
///@{
template <typename T>
inline void fml::comm::allreduce(int n, T *data, MPI_Op op) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Allreduce(MPI_IN_PLACE, data, n, type, op, _comm);
  check_ret(ret);
}
///@}



/**
  @brief Sum reduce operation across all processes in the MPI communicator.
  
  @param[in] n Number of elemends of 'data'.
  @param[in,out] data The data to reduce.
  @param[in] root The rank in the MPI communicator to receive the final answer.
 */
///@{
template <typename T>
inline void fml::comm::reduce(int n, T *data, MPI_Op op, int root) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Reduce(MPI_IN_PLACE, data, n, type, op, root, _comm);
  check_ret(ret);
}
///@}



/**
  @brief Broadcast.
  
  @param[in] n Number of elemends of 'data'.
  @param[in,out] data The data to broadcast.
  @param[in] root The rank in the MPI communicator that does the broadcasting.
 */
///@{
template <typename T>
inline void fml::comm::bcast(int n, T *data, int root) const
{
  MPI_Datatype type = mpi_type_lookup(data);
  int ret = MPI_Bcast(data, n, type, root, _comm);
  check_ret(ret);
}
///@}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

inline void fml::comm::init()
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



inline void fml::comm::set_metadata()
{
  int ret;
  
  ret = MPI_Comm_rank(_comm, &_rank);
  check_ret(ret);
  
  ret = MPI_Comm_size(_comm, &_size);
  check_ret(ret);
}



inline void fml::comm::check_ret(const int ret) const
{
  if (ret != MPI_SUCCESS && _rank == 0)
  {
    int slen;
    char s[MPI_MAX_ERROR_STRING];
    
    MPI_Error_string(ret, s, &slen);
    throw std::runtime_error(s);
  }
}



template <typename T>
inline MPI_Datatype fml::comm::mpi_type_lookup(const T *x) const
{
  (void) x;
  
  // C types
  if (typeid(T) == typeid(char))
    return MPI_CHAR;
  else if (typeid(T) == typeid(double))
    return MPI_DOUBLE;
  else if (typeid(T) == typeid(float))
    return MPI_FLOAT;
  else if (typeid(T) == typeid(int))
    return MPI_INT;
  else if (typeid(T) == typeid(long))
    return MPI_LONG;
  else if (typeid(T) == typeid(long double))
    return MPI_LONG_DOUBLE;
  else if (typeid(T) == typeid(long long))
    return MPI_LONG_LONG_INT;
  else if (typeid(T) == typeid(short))
    return MPI_SHORT;
  else if (typeid(T) == typeid(unsigned int))
    return MPI_UNSIGNED;
  else if (typeid(T) == typeid(unsigned char))
    return MPI_UNSIGNED_CHAR;
  else if (typeid(T) == typeid(unsigned long))
    return MPI_UNSIGNED_LONG;
  else if (typeid(T) == typeid(unsigned short))
    return MPI_UNSIGNED_SHORT;
  else if (typeid(T) == typeid(uint32_t))
    return MPI_UINT32_T;
  
  // stdint types
  else if (typeid(T) == typeid(int8_t))
    return MPI_INT8_T;
  else if (typeid(T) == typeid(int16_t))
    return MPI_INT16_T;
  else if (typeid(T) == typeid(int32_t))
    return MPI_INT32_T;
  else if (typeid(T) == typeid(int64_t))
    return MPI_INT64_T;
  else if (typeid(T) == typeid(uint8_t))
    return MPI_UINT8_T;
  else if (typeid(T) == typeid(uint16_t))
    return MPI_UINT16_T;
  else if (typeid(T) == typeid(uint32_t))
    return MPI_UINT32_T;
  else if (typeid(T) == typeid(uint64_t))
    return MPI_UINT64_T;
  
  else
    return MPI_DATATYPE_NULL;
}



#endif
