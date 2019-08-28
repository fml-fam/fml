#ifndef FML_CPU_CPUMAT_H
#define FML_CPU_CPUMAT_H


#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "../fmlutils.hh"
#include "../types.hh"
#include "../unimat.hh"


/**
 * @brief Matrix class for data held on a single CPU. 
 * @tparam REAL should be 'float' or 'double'.
 */
template <typename REAL>
class cpumat : public unimat<REAL>
{
  public:
    cpumat();
    cpumat(len_t nrows, len_t ncols);
    cpumat(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    cpumat(const cpumat &x);
    ~cpumat();
    
    void resize(len_t nrows, len_t ncols);
    void set(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    cpumat<REAL> dupe() const;
    
    void print(uint8_t ndigits=4) const;
    void info() const;
    
    void fill_zero();
    void fill_one();
    void fill_val(const REAL v);
    void fill_linspace(REAL min, REAL max);
    void fill_eye();
    void fill_runif(uint32_t seed, REAL min=0, REAL max=1);
    void fill_runif(REAL min=0, REAL max=1);
    void fill_rnorm(uint32_t seed, REAL mean=0, REAL sd=1);
    void fill_rnorm(REAL mean=0, REAL sd=1);
    void scale(const REAL s);
    
    REAL& operator()(len_t i);
    const REAL& operator()(len_t i) const;
    REAL& operator()(len_t i, len_t j);
    const REAL& operator()(len_t i, len_t j) const;
    
    bool operator==(const cpumat<REAL> &x) const;
    bool operator!=(const cpumat<REAL> &x) const;
  
  private:
    bool free_data;
    bool should_free() const {return free_data;};
    void free();
    void printval(const REAL val, uint8_t ndigits) const;
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

template <typename REAL>
cpumat<REAL>::cpumat()
{
  this->m = 0;
  this->n = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename REAL>
cpumat<REAL>::cpumat(len_t nrows, len_t ncols)
{
  size_t len = (size_t) nrows * ncols * sizeof(REAL);
  this->data = (REAL*) std::malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
  
  this->m = nrows;
  this->n = ncols;
  
  this->free_data = true;
}



template <typename REAL>
cpumat<REAL>::cpumat(REAL *data_, len_t nrows, len_t ncols, bool free_on_destruct)
{
  this->m = nrows;
  this->n = ncols;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
cpumat<REAL>::cpumat(const cpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->free_data = x.should_free();
}



template <typename REAL>
cpumat<REAL>::~cpumat()
{
  this->free();
}



// memory management

template <typename REAL>
void cpumat<REAL>::resize(len_t nrows, len_t ncols)
{
  size_t len = (size_t) nrows * ncols * sizeof(REAL);
  size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
  if (len == oldlen)
  {
    this->m = nrows;
    this->n = ncols;
    return;
  }
  
  void *realloc_ptr = realloc(this->data, len);
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (REAL*) realloc_ptr;
  
  this->m = nrows;
  this->n = ncols;
}



template <typename REAL>
void cpumat<REAL>::set(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct)
{
  this->free();
  
  this->m = nrows;
  this->n = ncols;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
cpumat<REAL> cpumat<REAL>::dupe() const
{
  cpumat<REAL> cpy(this->m, this->n);
  
  size_t len = (size_t) this->m * this->n * sizeof(REAL);
  memcpy(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

template <typename REAL>
void cpumat<REAL>::print(uint8_t ndigits) const
{
  for (len_t i=0; i<this->m; i++)
  {
    for (len_t j=0; j<this->n; j++)
      printval(this->data[i + this->m*j], ndigits);
    
    putchar('\n');
  }
  
  putchar('\n');
}



template <typename REAL>
void cpumat<REAL>::info() const
{
  printf("# cpumat");
  printf(" %dx%d", this->m, this->n);
  printf(" type=%s", typeid(REAL).name());
  printf("\n");
}



// fillers

template <typename REAL>
void cpumat<REAL>::fill_zero()
{
  size_t len = (size_t) this->m * this->n * sizeof(REAL);
  memset(this->data, 0, len);
}



template <typename REAL>
void cpumat<REAL>::fill_one()
{
  this->fill_val((REAL) 1);
}



template <typename REAL>
void cpumat<REAL>::fill_val(const REAL v)
{
  #pragma omp parallel for simd
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] = v;
  }
}



template <typename REAL>
void cpumat<REAL>::fill_linspace(REAL min, REAL max)
{
  if (min == max)
    this->fill_val(min);
  else
  {
    REAL v = (max-min)/((REAL) this->m*this->n - 1);
    for (len_t j=0; j<this->n; j++)
    {
      for (len_t i=0; i<this->m; i++)
      {
        len_t ind = i + this->m*j;
        this->data[ind] = v*((REAL) ind) + min;
      }
    }
  }
}



template <typename REAL>
void cpumat<REAL>::fill_eye()
{
  this->fill_zero();
  for (len_t i=0; i<this->m && i<this->n; i++)
    this->data[i + this->m*i] = (REAL) 1;
}



template <typename REAL>
void cpumat<REAL>::fill_runif(uint32_t seed, REAL min, REAL max)
{
  std::mt19937 mt(seed);
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      static std::uniform_real_distribution<REAL> dist(min, max);
      this->data[i + this->m*j] = dist(mt);
    }
  }
}

template <typename REAL>
void cpumat<REAL>::fill_runif(REAL min, REAL max)
{
  uint32_t seed = fmlutils::get_seed();
  this->fill_runif(seed, min, max);
}



template <typename REAL>
void cpumat<REAL>::fill_rnorm(uint32_t seed, REAL mean, REAL sd)
{
  std::mt19937 mt(seed);
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      static std::normal_distribution<REAL> dist(mean, sd);
      this->data[i + this->m*j] = dist(mt);
    }
  }
}

template <typename REAL>
void cpumat<REAL>::fill_rnorm(REAL mean, REAL sd)
{
  uint32_t seed = fmlutils::get_seed();
  this->fill_rnorm(seed, mean, sd);
}



template <typename REAL>
void cpumat<REAL>::scale(const REAL s)
{
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] *= s;
  }
}



// operators

template <typename REAL>
REAL& cpumat<REAL>::operator()(len_t i)
{
  if (i < 0 || i >= (this->m * this->n))
    throw std::runtime_error("index out of bounds");
  
  return this->data[i];
}

template <typename REAL>
const REAL& cpumat<REAL>::operator()(len_t i) const
{
  if (i < 0 || i >= (this->m * this->n))
    throw std::runtime_error("index out of bounds");
  
  return this->data[i];
}



template <typename REAL>
REAL& cpumat<REAL>::operator()(len_t i, len_t j)
{
  if (i < 0 || i >= this->m || j < 0 || j >= this->n)
    throw std::runtime_error("index out of bounds");
  
  return this->data[i + (this->m)*j];
}

template <typename REAL>
const REAL& cpumat<REAL>::operator()(len_t i, len_t j) const
{
  if (i < 0 || i >= this->m || j < 0 || j >= this->n)
    throw std::runtime_error("index out of bounds");
  
  return this->data[i + (this->m)*j];
}



template <typename REAL>
bool cpumat<REAL>::operator==(const cpumat<REAL> &x) const
{
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  
  if (this->data == x.data_ptr())
    return true;
  
  const REAL *x_d = x.data_ptr();
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      if (this->data[i + this->m*j] != x_d[i + this->m*j])
        return false;
    }
  }
  
  return true;
}

template <typename REAL>
bool cpumat<REAL>::operator!=(const cpumat<REAL> &x) const
{
  return !(*this == x);
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename REAL>
void cpumat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <>
inline void cpumat<int>::printval(const int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

template <typename REAL>
void cpumat<REAL>::printval(const REAL val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}



#endif
