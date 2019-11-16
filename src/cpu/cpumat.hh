#ifndef FML_CPU_CPUMAT_H
#define FML_CPU_CPUMAT_H


#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "../fmlutils.hh"
#include "../omputils.hh"
#include "../types.hh"
#include "../unimat.hh"

#include "cpuvec.hh"


/**
 * @brief Matrix class for data held on a single CPU.
 * 
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
    
    void print(uint8_t ndigits=4, bool add_final_blank=true) const;
    void info() const;
    
    void fill_zero();
    void fill_one();
    void fill_val(const REAL v);
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_diag(const cpuvec<REAL> &v);
    void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
    void fill_runif(const REAL min=0, const REAL max=1);
    void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
    void fill_rnorm(const REAL mean=0, const REAL sd=1);
    
    void diag(cpuvec<REAL> &v);
    void antidiag(cpuvec<REAL> &v);
    void scale(const REAL s);
    void rev_rows();
    void rev_cols();
    void get_row(len_t i, cpuvec<REAL> &v) const;
    void get_col(len_t j, cpuvec<REAL> &v) const;
    
    bool any_inf() const;
    bool any_nan() const;
    
    const REAL operator()(len_t i) const; // getters
    const REAL operator()(len_t i, len_t j) const;
    REAL& operator()(len_t i); // setters
    REAL& operator()(len_t i, len_t j);
    
    bool operator==(const cpumat<REAL> &x) const;
    bool operator!=(const cpumat<REAL> &x) const;
    
    cpumat<REAL>& operator=(const cpumat<REAL> &x);
  
  private:
    void free();
    void check_params(len_t nrows, len_t ncols);
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
  check_params(nrows, ncols);
  
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
  check_params(nrows, ncols);
  
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
  
  this->free_data = false;
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
  check_params(nrows, ncols);
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  const size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
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
  check_params(nrows, ncols);
  
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
  
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  memcpy(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

template <typename REAL>
void cpumat<REAL>::print(uint8_t ndigits, bool add_final_blank) const
{
  for (len_t i=0; i<this->m; i++)
  {
    for (len_t j=0; j<this->n; j++)
      this->printval(this->data[i + this->m*j], ndigits);
    
    putchar('\n');
  }
  
  if (add_final_blank)
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
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
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
  #pragma omp parallel for if((this->m)*(this->n) > omputils::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
  {
    #pragma omp simd
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] = v;
  }
}



template <>
inline void cpumat<int>::fill_linspace(const int start, const int stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const float v = (stop-start)/((float) this->m*this->n - 1);
    
    #pragma omp parallel for if((this->m)*(this->n) > omputils::OMP_MIN_SIZE)
    for (len_t j=0; j<this->n; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<this->m; i++)
      {
        const len_t ind = i + this->m*j;
        this->data[ind] = (int) roundf(v*((float) ind) + start);
      }
    }
  }
}

template <typename REAL>
void cpumat<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const REAL v = (stop-start)/((REAL) this->m*this->n - 1);
    
    #pragma omp parallel for if((this->m)*(this->n) > omputils::OMP_MIN_SIZE)
    for (len_t j=0; j<this->n; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<this->m; i++)
      {
        const len_t ind = i + this->m*j;
        this->data[ind] = v*((REAL) ind) + start;
      }
    }
  }
}



template <typename REAL>
void cpumat<REAL>::fill_eye()
{
  cpuvec<REAL> v(1);
  v(0) = (REAL) 1;
  this->fill_diag(v);
}



template <typename REAL>
void cpumat<REAL>::fill_diag(const cpuvec<REAL> &v)
{
  this->fill_zero();
  
  REAL *v_d = v.data_ptr();
  len_t min = std::min(this->m, this->n);
  
  #pragma omp for simd
  for (len_t i=0; i<min; i++)
    this->data[i + this->m*i] = v_d[i % v.size()];
}



template <typename REAL>
void cpumat<REAL>::fill_runif(const uint32_t seed, const REAL min, const REAL max)
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
void cpumat<REAL>::fill_runif(const REAL min, const REAL max)
{
  uint32_t seed = fmlutils::get_seed();
  this->fill_runif(seed, min, max);
}



template <typename REAL>
void cpumat<REAL>::fill_rnorm(const uint32_t seed, const REAL mean, const REAL sd)
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
void cpumat<REAL>::fill_rnorm(const REAL mean, const REAL sd)
{
  uint32_t seed = fmlutils::get_seed();
  this->fill_rnorm(seed, mean, sd);
}



template <typename REAL>
void cpumat<REAL>::diag(cpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp for simd
  for (len_t i=0; i<minmn; i++)
    v_ptr[i] = this->data[i + this->m*i];
}



template <typename REAL>
void cpumat<REAL>::antidiag(cpuvec<REAL> &v)
{
  const len_t minmn = std::min(this->m, this->n);
  v.resize(minmn);
  REAL *v_ptr = v.data_ptr();
  
  #pragma omp for simd
  for (len_t i=0; i<minmn; i++)
    v_ptr[i] = this->data[(this->m-1-i) + this->m*i];
}



template <typename REAL>
void cpumat<REAL>::scale(const REAL s)
{
  #pragma omp parallel for if((this->m)*(this->n) > omputils::OMP_MIN_SIZE)
  for (len_t j=0; j<this->n; j++)
  {
    #pragma omp simd
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] *= s;
  }
}



template <typename REAL>
void cpumat<REAL>::rev_rows()
{
  for (len_t j=0; j<this->n; j++)
  {
    len_t last = this->m - 1;
    for (len_t i=0; i<this->m/2; i++)
    {
      const REAL tmp = this->data[i + this->m*j];
      this->data[i + this->m*j] = this->data[last + this->m*j];
      this->data[last + this->m*j] = tmp;
    }
    
    last--;
  }
}



template <typename REAL>
void cpumat<REAL>::rev_cols()
{
  len_t last = this->n - 1;
  
  for (len_t j=0; j<this->n/2; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      const REAL tmp = this->data[i + this->m*j];
      this->data[i + this->m*j] = this->data[i + this->m*last];
      this->data[i + this->m*last] = tmp;
    }
    
    last--;
  }
}



template <typename REAL>
void cpumat<REAL>::get_row(len_t i, cpuvec<REAL> &v) const
{
  v.resize(this->n);
  REAL *v_d = v.data_ptr();
  
  for (len_t j=0; j<this->n; j++)
    v_d[j] = this->data[i + this->m*j];
}



template <typename REAL>
void cpumat<REAL>::get_col(len_t j, cpuvec<REAL> &v) const
{
  v.resize(this->m);
  REAL *v_d = v.data_ptr();
  
  for (len_t i=0; i<this->m; i++)
    v_d[i] = this->data[i + this->m*j];
}



template <typename REAL>
bool cpumat<REAL>::any_inf() const
{
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      if (isinf(this->data[i + this->m*j]))
        return true;
    }
  }
  
  return false;
}



template <typename REAL>
bool cpumat<REAL>::any_nan() const
{
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
    {
      if (isnan(this->data[i + this->m*j]))
        return true;
    }
  }
  
  return false;
}



// operators

template <typename REAL>
const REAL cpumat<REAL>::operator()(len_t i) const
{
  this->check_index(i);
  return this->data[i];
}

template <typename REAL>
const REAL cpumat<REAL>::operator()(len_t i, len_t j) const
{
  this->check_index(i, j);
  return this->data[i + (this->m)*j];
}

template <typename REAL>
REAL& cpumat<REAL>::operator()(len_t i)
{
  this->check_index(i);
  return this->data[i];
}

template <typename REAL>
REAL& cpumat<REAL>::operator()(len_t i, len_t j)
{
  this->check_index(i, j);
  return this->data[i + (this->m)*j];
}



template <typename REAL>
bool cpumat<REAL>::operator==(const cpumat<REAL> &x) const
{
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  else if (this->data == x.data_ptr())
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



template <typename REAL>
cpumat<REAL>& cpumat<REAL>::operator=(const cpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->free_data = false;
  return *this;
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



template <typename REAL>
void cpumat<REAL>::check_params(len_t nrows, len_t ncols)
{
  if (nrows < 0 || ncols < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
