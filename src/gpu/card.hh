// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_CARD_H
#define FML_GPU_CARD_H
#pragma once


#include <memory>
#include <stdexcept>

#include "arch/arch.hh"


/**
  @brief GPU data and methods.
  
  @impl Stores GPU ordinal and BLAS/LAPACK handles. Methods are wrappers
  around core GPU operations, allowing GPU malloc, memset, etc.
  
  @details You probably should not use these methods directly unless you know
  what you are doing (in which case you probably do not even need them). Simply
  pass a card object to a GPU object constructor and move on.
 */
class card
{
  public:
    card();
    card(const int id=0);
    card(const card &x);
    ~card();
    
    void set(const int id);
    
    void info() const;
    
    void* mem_alloc(const size_t len);
    void mem_set(void *ptr, const int value, const size_t len);
    void mem_free(void *ptr);
    void mem_cpu2gpu(void *dst, const void *src, const size_t len);
    void mem_gpu2cpu(void *dst, const void *src, const size_t len);
    void mem_gpu2gpu(void *dst, const void *src, const size_t len);
    
    void synch();
    void check();
    
    ///@{
    /// The ordinal number corresponding to the GPU device.
    int get_id() {return _id;};
    int get_id() const {return _id;};
    /// GPU BLAS handle.
    gpublas_handle_t blas_handle() {return _blas_handle;};
    gpublas_handle_t blas_handle() const {return _blas_handle;};
    /// GPU LAPACK handle.
    gpulapack_handle_t lapack_handle() {return _lapack_handle;};
    gpulapack_handle_t lapack_handle() const {return _lapack_handle;};
    /// Is the gpu data valid?
    bool valid_card() const {return (_id!=UNINITIALIZED_CARD && _id!=DESTROYED_CARD);};
    ///@}
  
  protected:
    int _id;
    gpublas_handle_t _blas_handle;
    gpulapack_handle_t _lapack_handle;
  
  private:
    static const int UNINITIALIZED_CARD = -1;
    static const int DESTROYED_CARD = -11;
    
    void init();
    void cleanup();
    gpu_error_t err;
    void check_gpu_error();
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/// @brief Create a new card object. Does not initialize any GPU data.
inline card::card()
{
  _id = UNINITIALIZED_CARD;
  _blas_handle = NULL;
  _lapack_handle = NULL;
}



/**
  @brief Create a new card object and set up internal CUDA data.
  
  @details Sets the current device to the provided GPU id and initializes GPU
  BLAS and LAPACK handles.
  
  @param[in] id Ordinal number corresponding to the desired GPU device.
  
  @except If the GPU can not be initialized, or if the allocation of one of the
  handles fails, the method will throw a 'runtime_error' exception.
*/
inline card::card(const int id)
{
  _id = id;
  init();
  
  gpublas_status_t blas_status = gpuprims::gpu_blas_init(&_blas_handle);
  if (blas_status != GPUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize GPU BLAS");
  
  gpulapack_status_t lapack_status = gpuprims::gpu_lapack_init(&_lapack_handle);
  if (lapack_status != GPULAPACK_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize GPU LAPACK");
}



inline card::card(const card &x)
{
  _id = x.get_id();
  _blas_handle = x.blas_handle();
  _lapack_handle = x.lapack_handle();
}



inline card::~card()
{
  cleanup();
}



/**
  @brief Sets up the existing card object.
  
  @details For use with the no-argument constructor. Frees any existing GPU
  data already allocated and stored in the object. Misuse of this could lead to
  some seemingly strange errors.
  
  @param[in] id Ordinal number corresponding to the desired GPU device.
  
  @except If the GPU can not be initialized, or if the allocation of one of the
  handles fails, the method will throw a 'runtime_error' exception.
*/
inline void card::set(const int id)
{
  if (id == _id)
    return;
  
  cleanup();
  
  _id = id;
  init();
  
  gpublas_status_t blas_status = gpuprims::gpu_blas_init(&_blas_handle);
  if (blas_status != GPUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize GPU BLAS");
  
  gpulapack_status_t lapack_status = gpuprims::gpu_lapack_init(&_lapack_handle);
  if (lapack_status != GPULAPACK_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize GPU LAPACK");
}



// printers

/**
  @brief Print some brief information about the GPU.
  
  @impl Uses NVML.
*/
inline void card::info() const
{
  nvml::init();
  
  int version = nvml::system::get_cuda_driver_version();
  int version_major = version / 1000;
  int version_minor = (version % 1000) / 10;
  
  nvmlDevice_t device = nvml::device::get_handle_by_index(_id);
  std::string name = nvml::device::get_name(device);
  double mem_used, mem_total;
  nvml::device::get_memory_info(device, &mem_used, &mem_total);
  
  printf("## GPU %d ", _id);
  printf("(%s) ", name.c_str());
  printf("%.0f/%.0f MB ", mem_used/1024/1024, mem_total/1024/1024);
  printf("- CUDA %d.%d", version_major, version_minor);
  printf("\n\n");
  
  nvml::shutdown();
}



// gpu memory management

/**
  @brief Allocate device memory.
  
  @param[in] len Number of bytes of memory to allocate.
  @return Pointer to the newly allocated device memory.
  
  @impl Wrapper around GPU malloc, e.g. `cudaMalloc()`.
  
  @except If the allocation fails, this throws a 'runtime_error' exception.
*/
inline void* card::mem_alloc(const size_t len)
{
  init();
  void *ptr;
  err = gpuprims::gpu_malloc(&ptr, len);
  check_gpu_error();
  return ptr;
}



/**
  @brief Set device memory.
  
  @param[in,out] ptr On entrance, the already-allocated block of memory to set.
  On exit, blocks of length 'len' will be set to 'value'.
  @param[in] value The value to set.
  @param[in] len Number of bytes of the input 'ptr' to set to 'value'.
  @return Pointer to the newly allocated device memory.
  
  @impl Wrapper around GPU memset, e.g. `cudaMemset()`.
  
  @except If the function fails (e.g., being by given non-device memory), this
  throws a 'runtime_error' exception.
*/
inline void card::mem_set(void *ptr, const int value, const size_t len)
{
  init();
  err = gpuprims::gpu_memset(ptr, value, len);
  check_gpu_error();
}



/**
  @brief Free device memory.
  
  @param[in] ptr The device memory you want to un-allocate.
  
  @impl Wrapper around GPU free, e.g. `cudaFree()`.
  
  @except If the function fails (e.g., being by given non-device memory), this
  throws a 'runtime_error' exception.
*/
inline void card::mem_free(void *ptr)
{
  init();
  if (ptr)
  {
    err = gpuprims::gpu_free(ptr);
    check_gpu_error();
  }
}



/**
  @brief Copy host (CPU) data to device (GPU) memory.
  
  @param[in,out] dst The device memory you want to copy TO.
  @param[in] src The host memory you want to copy FROM.
  @param[in] len Number of bytes of each array to use.
  
  @impl Wrapper around GPU memcpy, e.g. `cudaMemcpy()`.
  
  @except If the function fails (e.g., being by improperly using device
  memory), this throws a 'runtime_error' exception.
*/
inline void card::mem_cpu2gpu(void *dst, const void *src, const size_t len)
{
  init();
  err = gpuprims::gpu_memcpy(dst, src, len, GPU_MEMCPY_HOST_TO_DEVICE);
  check_gpu_error();
}



/**
  @brief Copy device (GPU) data to host (CPU) memory.
  
  @param[in,out] dst The host memory you want to copy TO.
  @param[in] src The device memory you want to copy FROM.
  @param[in] len Number of bytes of each array to use.
  
  @impl Wrapper around GPU memcpy, e.g. `cudaMemcpy()`.
  
  @except If the function fails (e.g., being by improperly using device
  memory), this throws a 'runtime_error' exception.
*/
inline void card::mem_gpu2cpu(void *dst, const void *src, const size_t len)
{
  init();
  err = gpuprims::gpu_memcpy(dst, src, len, GPU_MEMCPY_DEVICE_TO_HOST);
  check_gpu_error();
}



/**
  @brief Copy device (GPU) data to other device (GPU) memory.
  
  @param[in,out] dst The device memory you want to copy TO.
  @param[in] src The device memory you want to copy FROM.
  @param[in] len Number of bytes of each array to use.
  
  @impl Wrapper around GPU memcpy, e.g. `cudaMemcpy()`.
  
  @except If the function fails (e.g., being by improperly using device
  memory), this throws a 'runtime_error' exception.
*/
inline void card::mem_gpu2gpu(void *dst, const void *src, const size_t len)
{
  init();
  err = gpuprims::gpu_memcpy(dst, src, len, GPU_MEMCPY_DEVICE_TO_DEVICE);
  check_gpu_error();
}



/**
  @brief Copy device (GPU) data to other device (GPU) memory.
  
  @details Blocks further GPU execution until the device completes all
  previously executed kernels.
  
  @impl Wrapper around GPU synchronize, e.g. `cudaDeviceSynchronize()`.
  
  @except If a CUDA error is detected, this throws a 'runtime_error' exception.
*/
inline void card::synch()
{
  init();
  err = gpuprims::gpu_synch();
  check_gpu_error();
}



/**
  @brief Check for (and throw if found) a CUDA error.
  
  @impl Wrapper around GPU error lookup, e.g. `cudaGetLastError()`.
  
  @except If a CUDA error is detected, this throws a 'runtime_error' exception.
*/
inline void card::check()
{
  err = gpuprims::gpu_last_error();
  check_gpu_error();
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

inline void card::init()
{
  if (_id == UNINITIALIZED_CARD)
    throw std::runtime_error("invalid card (uninitialized)");
  else if (_id == DESTROYED_CARD)
    throw std::runtime_error("invalid card (destroyed)");
  
  err = gpuprims::gpu_set_device(_id);
  check_gpu_error();
}



inline void card::cleanup()
{
  init();
  
  if (_lapack_handle)
  {
    gpuprims::gpu_lapack_free(_lapack_handle);
    _lapack_handle = NULL;
  }
  
  if (_blas_handle)
  {
    gpuprims::gpu_blas_free(_blas_handle);
    _blas_handle = NULL;
  }
  
  err = gpuprims::gpu_device_reset();
  
  _id = DESTROYED_CARD;
}



inline void card::check_gpu_error()
{
  if (err != GPU_SUCCESS)
  {
    cleanup();
    
    std::string s = gpuprims::gpu_error_string(err);
    throw std::runtime_error(s);
  }
}


#endif
