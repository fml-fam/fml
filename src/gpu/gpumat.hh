#ifndef FML_GPU_GPUMAT_H
#define FML_GPU_GPUMAT_H


#include <cstdint>
#include <cstdio>

#include "../types.hh"
#include "../unimat.hh"

#include "card.hh"
#include "kernelfuns.hh"
#include "launcher.hh"
#include "gpuvec.hh"


/**
 * @brief Matrix class for data held on a single GPU. 
 * 
 * @tparam REAL should be '__half', 'float', or 'double'.
 */
template <typename REAL>
class gpumat : public unimat<REAL>
{
  public:
    gpumat(std::shared_ptr<card> gpu);
    gpumat(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    gpumat(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat(const gpumat &x);
    ~gpumat();
    
    void resize(len_t nrows, len_t ncols);
    void resize(std::shared_ptr<card> gpu, len_t nrows, len_t ncols);
    void set(std::shared_ptr<card> gpu, REAL *data, len_t nrows, len_t ncols, bool free_on_destruct=false);
    gpumat<REAL> dupe() const;
    
    void print(uint8_t ndigits=4, bool add_final_blank=true) const;
    void info() const;
    
    void fill_zero();
    void fill_one();
    void fill_val(const REAL v);
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_diag(const gpuvec<REAL> &v);
    void fill_runif(const uint32_t seed, const REAL min=0, const REAL max=1);
    void fill_runif(const REAL min=0, const REAL max=1);
    void fill_rnorm(const uint32_t seed, const REAL mean=0, const REAL sd=1);
    void fill_rnorm(const REAL mean=0, const REAL sd=1);
    
    void scale(const REAL s);
    void rev_rows();
    void rev_cols();
    
    bool any_inf() const;
    bool any_nan() const;
    
    const REAL operator()(len_t i) const; // getters
    const REAL operator()(len_t i, len_t j) const;
    // REAL& operator()(len_t i); // setters
    // REAL& operator()(len_t i, len_t j);
    
    bool operator==(const gpumat<REAL> &x) const;
    bool operator!=(const gpumat<REAL> &x) const;
    
    gpumat<REAL>& operator=(const gpumat<REAL> &x);
    
    std::shared_ptr<card> get_card() const {return c;};
    dim3 get_blockdim() const {return dim_block;};
    dim3 get_griddim() const {return dim_grid;};
    
  protected:
    std::shared_ptr<card> c;
  
  private:
    void free();
    void check_params(len_t nrows, len_t ncols);
    dim3 dim_block;
    dim3 dim_grid;
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
  check_params(nrows, ncols);
  
  this->c = gpu;
  
  const size_t len = (size_t) nrows * ncols * sizeof(REAL);
  this->data = (REAL*) this->c->mem_alloc(len);
  
  this->m = nrows;
  this->n = ncols;
  
  dim_block = kernel_launcher::dim_block2();
  dim_grid = kernel_launcher::dim_grid(this->m, this->n);
  
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
  
  dim_block = kernel_launcher::dim_block2();
  dim_grid = kernel_launcher::dim_grid(this->m, this->n);
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
gpumat<REAL>::gpumat(const gpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
  
  dim_block = kernel_launcher::dim_block2();
  dim_grid = kernel_launcher::dim_grid(this->m, this->n);
  
  this->c = x.get_card();
  
  this->free_data = false;
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
  
  dim_block = kernel_launcher::dim_block2();
  dim_grid = kernel_launcher::dim_grid(this->m, this->n);
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
  
  dim_block = kernel_launcher::dim_block2();
  dim_grid = kernel_launcher::dim_grid(this->m, this->n);
  
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
void gpumat<REAL>::print(uint8_t ndigits, bool add_final_blank) const
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
  
  if (add_final_blank)
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
  kernelfuns::kernel_fill_val<<<dim_grid, dim_block>>>(v, this->m, this->n, this->data);
  this->c->check();
}



template <typename REAL>
void gpumat<REAL>::fill_linspace(REAL start, REAL stop)
{
  // if (start == stop)
  //   this->fill_val(start);
  // else
  {
    kernelfuns::kernel_fill_linspace<<<dim_grid, dim_block>>>(start, stop, this->m, this->n, this->data);
    this->c->check();
  }
}




template <typename REAL>
void gpumat<REAL>::fill_eye()
{
  kernelfuns::kernel_fill_eye<<<dim_grid, dim_block>>>(this->m, this->n, this->data);
  this->c->check();
}



template <typename REAL>
void gpumat<REAL>::fill_diag(const gpuvec<REAL> &v)
{
  kernelfuns::kernel_fill_diag<<<dim_grid, dim_block>>>(v.size(), v.data_ptr(), this->m, this->n, this->data);
  this->c->check();
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
void diag(cpuvec<REAL> &v)
{
  
}

template <typename REAL>
void antidiag(cpuvec<REAL> &v)
{
  
}



template <typename REAL>
void gpumat<REAL>::scale(const REAL s)
{
  kernelfuns::kernel_scale<<<dim_grid, dim_block>>>(s, this->m, this->n, this->data);
  this->c->check();
}



template <typename REAL>
void gpumat<REAL>::rev_rows()
{
  kernelfuns::kernel_rev_rows<<<dim_grid, dim_block>>>(this->m, this->n, this->data);
  this->c->check();
}



template <typename REAL>
void gpumat<REAL>::rev_cols()
{
  kernelfuns::kernel_rev_cols<<<dim_grid, dim_block>>>(this->m, this->n, this->data);
  this->c->check();
}



template <typename REAL>
bool gpumat<REAL>::any_inf() const
{
  int has_inf = 0;
  int *has_inf_gpu = (int*) this->c->mem_alloc(sizeof(*has_inf_gpu));
  this->c->mem_cpu2gpu(has_inf_gpu, &has_inf, sizeof(has_inf));
  
  kernelfuns::kernel_any_inf<<<dim_grid, dim_block>>>(this->m, this->n, this->data, has_inf_gpu);
  
  this->c->mem_gpu2cpu(&has_inf, has_inf_gpu, sizeof(has_inf));
  this->c->mem_free(has_inf_gpu);
  
  this->c->check();
  
  return (bool) has_inf;
}



template <typename REAL>
bool gpumat<REAL>::any_nan() const
{
  int has_nan = 0;
  int *has_nan_gpu = (int*) this->c->mem_alloc(sizeof(*has_nan_gpu));
  this->c->mem_cpu2gpu(has_nan_gpu, &has_nan, sizeof(has_nan));
  
  kernelfuns::kernel_any_nan<<<dim_grid, dim_block>>>(this->m, this->n, this->data, has_nan_gpu);
  
  this->c->mem_gpu2cpu(&has_nan, has_nan_gpu, sizeof(has_nan));
  this->c->mem_free(has_nan_gpu);
  
  this->c->check();
  
  return (bool) has_nan;
}



// operators

template <typename REAL>
const REAL gpumat<REAL>::operator()(len_t i) const
{
  this->check_index(i);
  
  REAL ret;
  this->c->mem_gpu2cpu(&ret, this->data + i, sizeof(REAL));
  return ret;
}

template <typename REAL>
const REAL gpumat<REAL>::operator()(len_t i, len_t j) const
{
  this->check_index(i, j);
  
  REAL ret;
  this->c->mem_gpu2cpu(&ret, this->data + (i + this->m*j), sizeof(REAL));
  return ret;
}

// template <typename REAL>
// REAL& gpumat<REAL>::operator()(len_t i)
// {
//   this->check_index(i);
// 
// }
// 
// template <typename REAL>
// REAL& gpumat<REAL>::operator()(len_t i, len_t j)
// {
//   this->check_index(i, j);
// 
// }



template <typename T>
bool gpumat<T>::operator==(const gpumat<T> &x) const
{
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  else if (this->c->get_id() != x.get_card()->get_id())
    return false;
  else if (this->data == x.data_ptr())
    return true;
  
  int all_eq = 1;
  int *all_eq_gpu = (int*) this->c->mem_alloc(sizeof(*all_eq_gpu));
  this->c->mem_cpu2gpu(all_eq_gpu, &all_eq, sizeof(all_eq));
  
  kernelfuns::kernel_all_eq<<<dim_grid, dim_block>>>(this->m, this->n, this->data, x.data_ptr(), all_eq_gpu);
  
  this->c->mem_gpu2cpu(&all_eq, all_eq_gpu, sizeof(all_eq));
  this->c->mem_free(all_eq_gpu);
  
  this->c->check();
  
  return (bool) all_eq;
}

template <typename T>
bool gpumat<T>::operator!=(const gpumat<T> &x) const
{
  return !(*this == x);
}



template <typename REAL>
gpumat<REAL>& gpumat<REAL>::operator=(const gpumat<REAL> &x)
{
  this->c = x.get_card();
  
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
void gpumat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    this->c->mem_free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void gpumat<REAL>::check_params(len_t nrows, len_t ncols)
{
  if (nrows < 0 || ncols < 0)
    throw std::runtime_error("invalid dimensions");
}


#endif
