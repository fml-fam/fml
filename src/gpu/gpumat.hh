#ifndef FML_GPU_GPUMAT_H
#define FML_GPU_GPUMAT_H


#include <cstdint>
#include <cstdio>

#include "card.hh"
#include "kernelfuns.hh"

#include "../types.hh"
#include "../unimat.hh"


template <typename REAL>
class gpumat : public unimat<REAL>
{
  public:
    gpumat();
    gpumat(std::shared_ptr<card> gpu);
    gpumat(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    gpumat(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat(const gpumat &x);
    ~gpumat();
    
    void resize(len_t nrows, len_t ncols);
    void resize(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    void set(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat<REAL> dupe() const;
    
    void print(uint8_t ndigits=4) const;
    void info() const;
    
    void fill_zero();
    void fill_one();
    void fill_val(const REAL v);
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
    void fill_runif(const REAL min=0, const REAL max=1);
    void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
    void fill_rnorm(const REAL mean=0, const REAL sd=1);
    void scale(const REAL s);
    
    REAL& operator()(len_t i);
    const REAL& operator()(len_t i) const;
    REAL& operator()(len_t i, len_t j);
    const REAL& operator()(len_t i, len_t j) const;
    
    bool operator==(const gpumat<REAL> &x) const;
    bool operator!=(const gpumat<REAL> &x) const;
    
    std::shared_ptr<card> get_card() const {return c;};
  
  private:
    std::shared_ptr<card> c;
    bool free_data;
    bool should_free() const {return free_data;};
    void free();
    void printval(const REAL val, uint8_t ndigits) const;
    void check_params(len_t nrows, len_t ncols);
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

template <typename REAL>
gpumat<REAL>::gpumat()
{
  std::shared_ptr<card> gpu;
  this->c = gpu;
  
  this->m = 0;
  this->n = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu)
{
  this->c = gpu;
  
  this->m = 0;
  this->n = 0;
  this->data = NULL;
  
  this->free_data = true;
}



template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu, len_t nrows, len_t ncols)
{
  check_params(nrows, ncols);
  
  this->c = gpu;
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  this->data = (REAL*) this->c->mem_alloc(len);
  
  this->m = nrows;
  this->n = ncols;
  
  this->free_data = true;
}



template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu, REAL *data_, len_t nrows, len_t ncols, bool free_on_destruct)
{
  check_params(nrows, ncols);
  
  this->c = gpu;
  
  this->m = nrows;
  this->n = ncols;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
gpumat<REAL>::gpumat(const gpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  this->c = x.get_card();
  
  this->free_data = x.should_free();
}



template <typename REAL>
gpumat<REAL>::~gpumat()
{
  this->free();
  c = NULL;
}



// memory management

template <typename REAL>
void gpumat<REAL>::resize(len_t nrows, len_t ncols)
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
  
  REAL *realloc_ptr;
  realloc_ptr = (REAL*) this->c->mem_alloc(len);
  
  const size_t copylen = std::min(len, oldlen);
  this->c->mem_gpu2gpu(realloc_ptr, this->data, copylen);
  this->c->mem_free(this->data);
  this->data = realloc_ptr;
  
  this->m = nrows;
  this->n = ncols;
}



template <typename REAL>
void gpumat<REAL>::resize(std::shared_ptr<card> gpu, len_t nrows, len_t ncols)
{
  this->c = gpu;
  this->resize(nrows, ncols);
}



template <typename REAL>
void gpumat<REAL>::set(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct)
{
  check_params(nrows, ncols);
  
  this->free();
  
  this->c = gpu;
  
  this->m = nrows;
  this->n = ncols;
  this->data = data;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
gpumat<REAL> gpumat<REAL>::dupe() const
{
  gpumat<REAL> cpy(this->c, this->m, this->n);
  
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  this->c->mem_gpu2gpu(cpy.data_ptr(), this->data, len);
  
  return cpy;
}



// printers

template <typename REAL>
void gpumat<REAL>::print(uint8_t ndigits) const
{
  for (int i=0; i<this->m; i++)
  {
    for (int j=0; j<this->n; j++)
    {
      REAL tmp;
      this->c->mem_gpu2cpu(&tmp, this->data + (i + this->m*j), sizeof(REAL));
      this->printval(tmp, ndigits);
    }
  
    putchar('\n');
  }
  
  putchar('\n');
}



template <typename REAL>
void gpumat<REAL>::info() const
{
  printf("# gpumat ");
  printf("%dx%d ", this->m, this->n);
  printf("type=%s ", typeid(REAL).name());
  printf("\n");
}



// fillers

template <typename REAL>
void gpumat<REAL>::fill_zero()
{
  const size_t len = (size_t) this->m * this->n * sizeof(REAL);
  this->c->mem_set(this->data, 0, len);
}



template <typename REAL>
void gpumat<REAL>::fill_one()
{
  this->fill_val((REAL) 1);
}



template <typename REAL>
void gpumat<REAL>::fill_val(const REAL v)
{
  
}



template <typename REAL>
void gpumat<REAL>::fill_linspace(REAL start, REAL stop)
{
  // if (start == stop)
  //   this->fill_val(start);
  // else
  {
    
  }
}




template <typename REAL>
void gpumat<REAL>::fill_eye()
{
  
}



template <typename REAL>
void gpumat<REAL>::fill_runif(uint32_t seed, REAL min, REAL max)
{
  
}

template <typename REAL>
void gpumat<REAL>::fill_runif(REAL min, REAL max)
{
  
}

template <typename REAL>
void gpumat<REAL>::fill_rnorm(uint32_t seed, REAL mean, REAL sd)
{
  
}

template <typename REAL>
void gpumat<REAL>::fill_rnorm(REAL mean, REAL sd)
{
  
}




template <typename REAL>
void gpumat<REAL>::scale(const REAL s)
{
  
}



// operators




// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename REAL>
void gpumat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    this->c->mem_free(this->data);
    this->data = NULL;
  }
}



template <>
inline void gpumat<int>::printval(const int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

template <>
inline void gpumat<__half>::printval(const __half val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, (float)val);
}

template <typename REAL>
void gpumat<REAL>::printval(const REAL val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}



template <typename REAL>
void gpumat<REAL>::check_params(len_t nrows, len_t ncols)
{
  if (nrows < 0 || ncols < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
