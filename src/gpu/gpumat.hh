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
    gpumat(std::shared_ptr<card> gpu);
    gpumat(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    gpumat(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    ~gpumat();
    
    void resize(len_t nrows, len_t ncols);
    void set(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat<REAL> dupe() const;
    
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
    
    bool operator==(const gpumat<REAL> &x) const;
    bool operator!=(const gpumat<REAL> &x) const;
    
    std::shared_ptr<card> get_card() const {return c;};
  
  private:
    std::shared_ptr<card> c;
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
  this->c = gpu;
  
  size_t len = nrows * ncols * sizeof(REAL);
  this->data = (REAL*) this->c->mem_alloc(len);
  
  this->m = nrows;
  this->n = ncols;
  
  this->free_data = true;
}



template <typename REAL>
gpumat<REAL>::gpumat(std::shared_ptr<card> gpu, REAL *data_, len_t nrows, len_t ncols, bool free_on_destruct)
{
  this->c = gpu;
  
  this->m = nrows;
  this->n = ncols;
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
gpumat<REAL>::~gpumat()
{
  if (this->free_data)
    this->c->mem_free(this->data);
}



// memory management

template <typename REAL>
void gpumat<REAL>::resize(len_t nrows, len_t ncols)
{
  if ( (this->m == nrows || this->n == nrows) && (this->m == ncols || this->n == ncols) )
  {
    this->m = nrows;
    this->n = ncols;
    return;
  }
  
  size_t len = nrows * ncols * sizeof(REAL);
  
  REAL *realloc_ptr;
  realloc_ptr = (REAL*) this->c->mem_alloc(len);
  
  size_t oldlen = this->m * this->n * sizeof(REAL);
  size_t copylen = std::min(len, oldlen);
  this->c->mem_gpu2gpu(realloc_ptr, this->data, copylen);
  this->c->mem_free(this->data);
  this->data = realloc_ptr;
  
  this->m = nrows;
  this->n = ncols;
}



template <typename REAL>
void gpumat<REAL>::set(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct)
{
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
  
  size_t len = this->m * this->n * sizeof(REAL);
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
      cudaMemcpy(&tmp, this->data + (i + this->m*j), sizeof(REAL), cudaMemcpyDeviceToHost);
      printf("%.*f ", ndigits, tmp);
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
  size_t len = (this->m) * (this->n) * sizeof(REAL);
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
void gpumat<REAL>::fill_linspace(REAL min, REAL max)
{
  if (min == max)
    this->fill_val(min);
  else
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



#endif
