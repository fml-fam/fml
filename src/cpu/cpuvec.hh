#ifndef FML_CPU_CPUVEC_H
#define FML_CPU_CPUVEC_H


#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "../univec.hh"


/**
 * @brief Vector class for data held on a single CPU. 
 * @tparam T should be 'int', 'float' or 'double'.
 */
template <typename T>
class cpuvec : public univec<T>
{
  public:
    cpuvec();
    cpuvec(len_t size);
    cpuvec(T *data, len_t size, bool free_on_destruct=false);
    cpuvec(const cpuvec &x);
    ~cpuvec();
    
    void resize(len_t size);
    void set(T *data, len_t size, bool free_on_destruct=false);
    cpuvec<T> dupe() const;
    
    void print(uint8_t ndigits=4) const;
    void info() const;
    
    void fill_zero();
    void fill_one();
    void fill_val(const T v);
    void scale(const T s);
    
    T& operator()(len_t i);
    const T& operator()(len_t i) const;
    
    bool operator==(const cpuvec<T> &x) const;
    bool operator!=(const cpuvec<T> &x) const;
  
  private:
    bool free_data;
    bool should_free() const {return free_data;};
    void free();
    void printval(const T val, uint8_t ndigits) const;
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

template <typename T>
cpuvec<T>::cpuvec()
{
  this->len = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename T>
cpuvec<T>::cpuvec(len_t size)
{
  size_t len = size * sizeof(T);
  this->data = (T*) std::malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
  
  this->len = size;
  
  this->free_data = true;
}



template <typename T>
cpuvec<T>::cpuvec(T *data_, len_t size, bool free_on_destruct)
{
  this->len = size;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename T>
cpuvec<T>::cpuvec(const cpuvec<T> &x)
{
  this->len = x.size();
  this->data = x.data_ptr();
  
  this->free_data = x.should_free();
}



template <typename T>
cpuvec<T>::~cpuvec()
{
  this->free();
}



// memory management

template <typename T>
void cpuvec<T>::resize(len_t size)
{
  if (this->len == size)
    return;
  
  size_t len = size * sizeof(T);
  
  void *realloc_ptr = realloc(this->data, len);
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (T*) realloc_ptr;
  
  this->len = size;
}



template <typename T>
void cpuvec<T>::set(T *data, len_t size, bool free_on_destruct)
{
  this->free();
  
  this->len = size;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



template <typename T>
cpuvec<T> cpuvec<T>::dupe() const
{
  cpuvec<T> cpy(this->len);
  
  size_t len = this->len * sizeof(T);
  memcpy(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

template <typename T>
void cpuvec<T>::print(uint8_t ndigits) const
{
  for (len_t i=0; i<this->len; i++)
    printval(this->data[i], ndigits);
  
  printf("\n\n");
}



template <typename T>
void cpuvec<T>::info() const
{
  printf("# cpuvec");
  printf(" %d", this->len);
  printf(" type=%s", typeid(T).name());
  printf("\n");
}



// fillers

template <typename T>
void cpuvec<T>::fill_zero()
{
  size_t len = this->len * sizeof(T);
  memset(this->data, 0, len);
}



template <typename T>
void cpuvec<T>::fill_one()
{
  this->fill_val((T) 1);
}



template <typename T>
void cpuvec<T>::fill_val(const T v)
{
  #pragma omp parallel for simd
  for (len_t i=0; i<this->len; i++)
    this->data[i] = v;
}



template <typename T>
void cpuvec<T>::scale(const T s)
{
  for (len_t i=0; i<this->len; i++)
    this->data[i] *= s;
}



// operators

template <typename T>
T& cpuvec<T>::operator()(len_t i)
{
  if (i < 0 || i >= this->len)
    throw std::runtime_error("index out of bounds");
  
  return this->data[i];
}

template <typename T>
const T& cpuvec<T>::operator()(len_t i) const
{
  if (i < 0 || i >= this->len)
    throw std::runtime_error("index out of bounds");
  
  return this->data[i];
}



template <typename T>
bool cpuvec<T>::operator==(const cpuvec<T> &x) const
{
  if (this->len != x.size())
    return false;
  
  if (this->data == x.data_ptr())
    return true;
  
  const T *x_d = x.data_ptr();
  for (len_t i=0; i<this->len; i++)
  {
    if (this->data[i] != x_d[i])
      return false;
  }
  
  return true;
}

template <typename T>
bool cpuvec<T>::operator!=(const cpuvec<T> &x) const
{
  return !(*this == x);
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename T>
void cpuvec<T>::free()
{
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <>
inline void cpuvec<int>::printval(const int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

template <typename T>
void cpuvec<T>::printval(const T val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}



#endif
