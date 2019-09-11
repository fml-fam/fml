#ifndef FML_GPU_GPUVEC_H
#define FML_GPU_GPUVEC_H


#include <cstdint>
#include <cstdio>

#include "../univec.hh"
#include "card.hh"
#include "kernelfuns.hh"


template <typename T>
class gpuvec : public univec<T>
{
  public:
    gpuvec();
    gpuvec(std::shared_ptr<card> gpu);
    gpuvec(std::shared_ptr<card> gpu, len_t size);
    gpuvec(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct=false);
    ~gpuvec();
    
    void resize(len_t size);
    void resize(std::shared_ptr<card> gpu, len_t size);
    void set(std::shared_ptr<card> gpu);
    void set(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct=false);
    gpuvec<T> dupe() const;
    
    void print(uint8_t ndigits=4) const;
    void info() const;
    
    void fill_zero();
    void fill_one();
    void fill_val(const T v);
    void fill_linspace(const T start, const T stop);
    
    void scale(const T s);
    void rev();
    
    T& operator()(len_t i);
    const T& operator()(len_t i) const;
    
    bool operator==(const gpuvec<T> &x) const;
    bool operator!=(const gpuvec<T> &x) const;
    
    std::shared_ptr<card> get_card() const {return c;};
    
  protected:
    std::shared_ptr<card> c;
  
  private:
    void free();
    void printval(const T val, uint8_t ndigits) const;
    void check_params(len_t size);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

template <typename REAL>
gpuvec<REAL>::gpuvec()
{
  std::shared_ptr<card> gpu;
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu)
{
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu, len_t size)
{
  check_params(size);
  
  this->c = gpu;
  
  size_t len = (size_t) size * sizeof(T);
  this->data = (T*) this->c->mem_alloc(len);
  
  this->_size = size;
  
  this->free_data = true;
}



template <typename T>
gpuvec<T>::gpuvec(std::shared_ptr<card> gpu, T *data_, len_t size, bool free_on_destruct)
{
  check_params(size);
  
  this->c = gpu;
  
  this->_size = size;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename T>
gpuvec<T>::~gpuvec()
{
  this->free();
}



// memory management

template <typename T>
void gpuvec<T>::resize(len_t size)
{
  check_params(size);
  
  if (this->_size == size)
    return;
  
  size_t len = (size_t) size * sizeof(T);
  
  T *realloc_ptr;
  realloc_ptr = (T*) this->c->mem_alloc(len);
  
  size_t oldlen = (size_t) this->_size * sizeof(T);
  size_t copylen = std::min(len, oldlen);
  this->c->mem_gpu2gpu(realloc_ptr, this->data, copylen);
  this->c->mem_free(this->data);
  this->data = realloc_ptr;
  
  this->_size = size;
}



template <typename T>
void gpuvec<T>::resize(std::shared_ptr<card> gpu, len_t size)
{
  this->c = gpu;
  this->resize(size);
}



template <typename T>
void gpuvec<T>::set(std::shared_ptr<card> gpu)
{
  this->c = gpu;
  
  this->_size = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename T>
void gpuvec<T>::set(std::shared_ptr<card> gpu, T *data, len_t size, bool free_on_destruct)
{
  check_params(size);
  
  this->free();
  
  this->c = gpu;
  
  this->_size = size;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



template <typename T>
gpuvec<T> gpuvec<T>::dupe() const
{
  gpuvec<T> cpy(this->c, this->_size);
  
  size_t len = (size_t) this->_size * sizeof(T);
  this->c->mem_gpu2gpu(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

template <typename T>
void gpuvec<T>::print(uint8_t ndigits) const
{
  for (int i=0; i<this->_size; i++)
  {
    T tmp;
    this->c->mem_gpu2cpu(&tmp, this->data + i, sizeof(T));
    printf("%.*f ", ndigits, tmp);
  }
  
  printf("\n\n");
}



template <typename T>
void gpuvec<T>::info() const
{
  printf("# gpuvec ");
  printf("%d ", this->_size);
  printf("type=%s ", typeid(T).name());
  printf("\n");
}



// fillers

template <typename T>
void gpuvec<T>::fill_zero()
{
  size_t len = (size_t) this->_size * sizeof(T);
  this->c->mem_set(this->data, 0, len);
}



template <typename T>
void gpuvec<T>::fill_one()
{
  this->fill_val((T) 1);
}



template <typename T>
void gpuvec<T>::fill_val(const T v)
{
  
}



template <typename T>
void gpuvec<T>::fill_linspace(const T start, const T stop)
{
  
}



template <typename T>
void gpuvec<T>::scale(const T s)
{
  
}



template <typename T>
void gpuvec<T>::rev()
{
  kernelfuns::kernel_rev_vec<<<1, this->_size>>>(this->_size, this->data);
  this->c->check();
}



// operators




// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename REAL>
void gpuvec<REAL>::free()
{
  if (this->free_data && this->data)
  {
    this->c->mem_free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void gpuvec<REAL>::check_params(len_t size)
{
  if (size < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
