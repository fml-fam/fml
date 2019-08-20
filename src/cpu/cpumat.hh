#ifndef FML_CPU_CPUMAT_H
#define FML_CPU_CPUMAT_H


#include <cstdlib>
#include <cstring>
#include <random>

#include "../matrix.hh"


template <typename REAL>
class cpumat : public matrix<REAL>
{
  public:
    cpumat();
    cpumat(len_t nrows, len_t ncols);
    cpumat(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    cpumat(const cpumat &x);
    ~cpumat();
    
    void resize(len_t nrows, len_t ncols);
    void set(REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    cpumat<REAL> dupe();
    
    void print(uint8_t ndigits=4);
    void info();
    
    void fill_zero();
    void fill_one();
    void fill_val(const REAL v);
    void fill_eye();
    void fill_runif(int seed, REAL min=0, REAL max=1);
    void fill_rnorm(int seed, REAL mean=0, REAL sd=1);
    void scale(const REAL s);
  
  private:
    bool free_data;
    bool should_free() const {return free_data;};
    void printval(const REAL val, uint8_t ndigits);
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
  size_t len = nrows * ncols * sizeof(REAL);
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
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



// memory management

template <typename REAL>
void cpumat<REAL>::resize(len_t nrows, len_t ncols)
{
  if ( (this->m == nrows || this->n == nrows) && (this->m == ncols || this->n == ncols) )
  {
    this->m = nrows;
    this->n = ncols;
    return;
  }
  
  size_t len = nrows * ncols * sizeof(REAL);
  
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
  this->m = nrows;
  this->n = ncols;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
cpumat<REAL> cpumat<REAL>::dupe()
{
  cpumat<REAL> cpy(this->m, this->n);
  
  size_t len = this->m * this->n * sizeof(REAL);
  memcpy(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

template <typename REAL>
void cpumat<REAL>::print(uint8_t ndigits)
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
void cpumat<REAL>::info()
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
  size_t len = (this->m) * (this->n) * sizeof(REAL);
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
void cpumat<REAL>::fill_eye()
{
  this->fill_zero();
  for (len_t i=0; i<this->m && i<this->n; i++)
    this->data[i + this->m*i] = (REAL) 1;
}



template <typename REAL>
void cpumat<REAL>::fill_runif(int seed, REAL min, REAL max)
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
void cpumat<REAL>::fill_rnorm(int seed, REAL mean, REAL sd)
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
void cpumat<REAL>::scale(const REAL s)
{
  for (len_t j=0; j<this->n; j++)
  {
    for (len_t i=0; i<this->m; i++)
      this->data[i + this->m*j] *= s;
  }
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <>
void cpumat<int>::printval(const int val, uint8_t ndigits)
{
  (void)ndigits;
  printf("%d ", val);
}

template <typename REAL>
void cpumat<REAL>::printval(const REAL val, uint8_t ndigits)
{
  printf("%.*f ", ndigits, val);
}



#endif
