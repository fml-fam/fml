#ifndef FML_GPU_CARD_H
#define FML_GPU_CARD_H


#include <cublas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <memory>
#include <stdexcept>

#include "nvml.hh"


/**
 * @brief GPU data and methods.
 * 
 * @impl Stores GPU ordinal and cuBLAS/cuSOLVER handles. Methods are wrappers
   around core CUDA operations, like cudaMalloc(), cudaMemcpy(), etc.
 * 
 * @details You probably should not use these methods directly unless you know
   what you are doing (in which case you probably do not even need them). Simply
   pass a card object to a GPU object constructor and move on.
 */
class card
{
  public:
    card();
    card(int id=0);
    card(const card &x);
    ~card();
    
    void set(int id);
    
    void info() const;
    
    void* mem_alloc(size_t len);
    void mem_set(void *ptr, int value, size_t len);
    void mem_free(void *ptr);
    void mem_cpu2gpu(void *dst, void *src, size_t len);
    void mem_gpu2cpu(void *dst, void *src, size_t len);
    void mem_gpu2gpu(void *dst, void *src, size_t len);
    
    void synch();
    void check();
    
    ///@{
    /// The ordinal number corresponding to the GPU device.
    int get_id() {return _id;};
    int get_id() const {return _id;};
    /// cuBLAS handle.
    cublasHandle_t cb_handle() {return _cb_handle;};
    cublasHandle_t cb_handle() const {return _cb_handle;};
    /// cuSOLVER handle.
    cusolverDnHandle_t cs_handle() {return _cs_handle;};
    cusolverDnHandle_t cs_handle() const {return _cs_handle;};
    /// Is the gpu data valid?
    bool valid_card() const {return (_id!=UNINITIALIZED_CARD && _id!=DESTROYED_CARD);};
    ///@}
  
  protected:
    int _id;
    cublasHandle_t _cb_handle;
    cusolverDnHandle_t _cs_handle;
  
  private:
    static const int UNINITIALIZED_CARD = -1;
    static const int DESTROYED_CARD = -11;
    
    void init();
    void cleanup();
    cudaError_t cerr;
    void check_cuda_error();
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

/**
 * @brief Create a new card object. Does not initialize any GPU data.
*/
inline card::card()
{
  _id = UNINITIALIZED_CARD;
  _cb_handle = NULL;
  _cs_handle = NULL;
}



/**
 * @brief Create a new card object and set up internal CUDA data.
 * 
 * @details Sets the current device to the provided GPU id and initializes a
   cuBLAS handle and a cuSOLVER handle.
 * 
 * @param[in] id Ordinal number corresponding to the desired GPU device.
 * 
 * @except If the GPU can not be initialized, or if the allocation of one of the
   handles fails, the method will throw a 'runtime_error' exception.
*/
inline card::card(int id)
{
  _id = id;
  init();
  
  cublasStatus_t cb_status = cublasCreate(&_cb_handle);
  if (cb_status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize cuBLAS");
  
  cusolverStatus_t cs_status = cusolverDnCreate(&_cs_handle);
  if (cs_status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize cuSOLVER");
}



inline card::card(const card &x)
{
  _id = x.get_id();
  _cb_handle = x.cb_handle();
  _cs_handle = x.cs_handle();
}



inline card::~card()
{
  cleanup();
}



/**
 * @brief Sets up the existing card object.
 * 
 * @details For use with the no-argument constructor. Frees any existing GPU
   data already allocated and stored in the object. Misuse of this could lead to
   some seemingly strange errors.
 * 
 * @param[in] id Ordinal number corresponding to the desired GPU device.
 * 
 * @except If the GPU can not be initialized, or if the allocation of one of the
   handles fails, the method will throw a 'runtime_error' exception.
*/
inline void card::set(int id)
{
  if (id == _id)
    return;
  
  cleanup();
  
  _id = id;
  init();
  
  cublasStatus_t cb_status = cublasCreate(&_cb_handle);
  if (cb_status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize cuBLAS");
  
  cusolverStatus_t cs_status = cusolverDnCreate(&_cs_handle);
  if (cs_status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("unable to initialize cuSOLVER");
}



// printers

/**
 * @brief Print some brief information about the GPU.
 * 
 * @impl Uses NVML.
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
 * @brief Allocate device memory.
 * 
 * @param[in] len Number of bytes of memory to allocate.
 * @return Pointer to the newly allocated device memory.
 * 
 * @impl Wrapper around cudaMalloc().
 * 
 * @except If the allocation fails, this throws a 'runtime_error' exception.
*/
inline void* card::mem_alloc(size_t len)
{
  init();
  void *ptr;
  cerr = cudaMalloc(&ptr, len);
  check_cuda_error();
  return ptr;
}



/**
 * @brief Set device memory.
 * 
 * @param[in,out] ptr On entrance, the already-allocated block of memory to set.
   On exit, blocks of length 'len' will be set to 'value'.
 * @param[in] value The value to set.
 * @param[in] len Number of bytes of the input 'ptr' to set to 'value'.
 * @return Pointer to the newly allocated device memory.
 * 
 * @impl Wrapper around cudaMemset().
 * 
 * @except If the function fails (e.g., being by given non-device memory), this
   throws a 'runtime_error' exception.
*/
inline void card::mem_set(void *ptr, int value, size_t len)
{
  init();
  cerr = cudaMemset(ptr, value, len);
  check_cuda_error();
}



/**
 * @brief Free device memory.
 * 
 * @param[in] ptr The device memory you want to un-allocate.
 * 
 * @impl Wrapper around cudaFree().
 * 
 * @except If the function fails (e.g., being by given non-device memory), this
   throws a 'runtime_error' exception.
*/
inline void card::mem_free(void *ptr)
{
  init();
  if (ptr)
  {
    cerr = cudaFree(ptr);
    check_cuda_error();
  }
}



/**
 * @brief Copy host (CPU) data to device (GPU) memory.
 * 
 * @param[in,out] dst The device memory you want to copy TO.
 * @param[in] src The host memory you want to copy FROM.
 * @param[in] len Number of bytes of each array to use.
 * 
 * @impl Wrapper around cudaMemcpy().
 * 
 * @except If the function fails (e.g., being by improperly using device
   memory), this throws a 'runtime_error' exception.
*/
inline void card::mem_cpu2gpu(void *dst, void *src, size_t len)
{
  init();
  cerr = cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice);
  check_cuda_error();
}



/**
 * @brief Copy device (GPU) data to host (CPU) memory.
 * 
 * @param[in,out] dst The host memory you want to copy TO.
 * @param[in] src The device memory you want to copy FROM.
 * @param[in] len Number of bytes of each array to use.
 * 
 * @impl Wrapper around cudaMemcpy().
 * 
 * @except If the function fails (e.g., being by improperly using device
   memory), this throws a 'runtime_error' exception.
*/
inline void card::mem_gpu2cpu(void *dst, void *src, size_t len)
{
  init();
  cerr = cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost);
  check_cuda_error();
}



/**
 * @brief Copy device (GPU) data to other device (GPU) memory.
 * 
 * @param[in,out] dst The device memory you want to copy TO.
 * @param[in] src The device memory you want to copy FROM.
 * @param[in] len Number of bytes of each array to use.
 * 
 * @impl Wrapper around cudaMemcpy().
 * 
 * @except If the function fails (e.g., being by improperly using device
   memory), this throws a 'runtime_error' exception.
*/
inline void card::mem_gpu2gpu(void *dst, void *src, size_t len)
{
  init();
  cerr = cudaMemcpy(dst, src, len, cudaMemcpyDeviceToDevice);
  check_cuda_error();
}



/**
 * @brief Copy device (GPU) data to other device (GPU) memory.
 * 
 * @details Blocks further GPU execution until the device completes all
   previously executed kernels.
 * 
 * @impl Wrapper around cudaDeviceSynchronize().
 * 
 * @except If a CUDA error is detected, this throws a 'runtime_error' exception.
*/
inline void card::synch()
{
  init();
  cerr = cudaDeviceSynchronize();
  check_cuda_error();
}



/**
 * @brief Check for (and throw if found) a CUDA error.
 * 
 * @impl Wrapper around cudaGetLastError().
 * 
 * @except If a CUDA error is detected, this throws a 'runtime_error' exception.
*/
inline void card::check()
{
  cerr = cudaGetLastError();
  check_cuda_error();
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
  
  cerr = cudaSetDevice(_id);
  check_cuda_error();
}



inline void card::cleanup()
{
  init();
  
  if (_cs_handle)
  {
    cusolverDnDestroy(_cs_handle);
    _cs_handle = NULL;
  }
  
  if (_cb_handle)
  {
    cublasDestroy(_cb_handle);
    _cb_handle = NULL;
  }
  
  cerr = cudaDeviceReset();
  
  _id = DESTROYED_CARD;
}



inline void card::check_cuda_error()
{
  if (cerr != cudaSuccess)
  {
    cleanup();
    
    std::string s = cudaGetErrorString(cerr);
    throw std::runtime_error(s);
  }
}


#endif
