#ifndef FML_GPU_CARD_H
#define FML_GPU_CARD_H


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <memory>
#include <stdexcept>

#include "nvml.hh"


class card
{
  public:
    card(int id=0);
    ~card();
    
    void info() const;
    
    void* mem_alloc(size_t len);
    void mem_set(void *ptr, int value, size_t len);
    void mem_free(void *ptr);
    void mem_cpu2gpu(void *dst, void *src, size_t len);
    void mem_gpu2cpu(void *dst, void *src, size_t len);
    void mem_gpu2gpu(void *dst, void *src, size_t len);
    
    void synch();
    
    int device_id() const {return _id;};
    cublasHandle_t cb_handle() {return _cb_handle;};
    cusolverDnHandle_t cs_handle() {return _cs_handle;};
  
  protected:
    int _id;
    cublasHandle_t _cb_handle;
    cusolverDnHandle_t _cs_handle;
  
  private:
    void init();
    void cleanup();
    cudaError_t cerr;
    void check_cuda();
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

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



inline card::~card()
{
  cleanup();
}



// printers

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

inline void* card::mem_alloc(size_t len)
{
  init();
  void *ptr;
  cerr = cudaMalloc(&ptr, len);
  check_cuda();
  return ptr;
}



inline void card::mem_set(void *ptr, int value, size_t len)
{
  init();
  cerr = cudaMemset(ptr, value, len);
  check_cuda();
}



inline void card::mem_free(void *ptr)
{
  init();
  if (ptr)
  {
    cerr = cudaFree(ptr);
    check_cuda();
  }
}



inline void card::mem_cpu2gpu(void *dst, void *src, size_t len)
{
  init();
  cerr = cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice);
  check_cuda();
}



inline void card::mem_gpu2cpu(void *dst, void *src, size_t len)
{
  init();
  cerr = cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost);
  check_cuda();
}



inline void card::mem_gpu2gpu(void *dst, void *src, size_t len)
{
  init();
  cerr = cudaMemcpy(dst, src, len, cudaMemcpyDeviceToDevice);
  check_cuda();
}



inline void card::synch()
{
  init();
  cerr = cudaDeviceSynchronize();
  check_cuda();
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

inline void card::init()
{
  cerr = cudaSetDevice(_id);
  check_cuda();
}



inline void card::cleanup()
{
  init();
  
  if (_cs_handle)
    cusolverDnDestroy(_cs_handle);
  
  if (_cb_handle)
    cublasDestroy(_cb_handle);
  
  cerr = cudaSetDevice(_id);
  cerr = cudaDeviceReset();
}



inline void card::check_cuda()
{
  if (cerr != cudaSuccess)
  {
    cleanup();
    
    std::string s = cudaGetErrorString(cerr);
    throw std::runtime_error(s);
  }
}


#endif
