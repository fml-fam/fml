#ifndef FML_MPI_MPIMAT_H
#define FML_MPI_MPIMAT_H


#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

#include "grid.hh"
#include "bcutils.hh"

#include "../fmlutils.hh"
#include "../types.hh"
#include "../unimat.hh"


/**
 * @brief Matrix class for data distributed over MPI in the 2-d block cyclic
    format. 
 * 
 * @tparam REAL should be 'float' or 'double'.
 */
template <typename REAL>
class mpimat : public unimat<REAL>
{
  public:
    mpimat(grid &blacs_grid);
    mpimat(grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows=16, int bf_cols=16);
    mpimat(grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct=false);
    mpimat(const mpimat &x);
    ~mpimat();
    
    void resize(len_t nrows, len_t ncols, int bf_rows=16, int bf_cols=16);
    void set(grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct=false);
    mpimat<REAL> dupe() const;
    
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
    
    REAL operator()(len_t i);
    const REAL operator()(len_t i) const;
    REAL operator()(len_t i, len_t j);
    const REAL operator()(len_t i, len_t j) const;
    
    bool operator==(const mpimat<REAL> &x) const;
    bool operator!=(const mpimat<REAL> &x) const;
    
    len_local_t nrows_local() const {return m_local;};
    len_local_t ncols_local() const {return n_local;};
    int bf_rows() const {return mb;};
    int bf_cols() const {return nb;};
    int* desc_ptr() {return desc;};
    const int* desc_ptr() const {return desc;};
    const grid get_grid() const {return g;};
    
  protected:
    len_local_t m_local;
    len_local_t n_local;
    int mb;
    int nb;
    int desc[9];
    grid g;
    
  private:
    bool free_data;
    bool should_free() const {return free_data;};
    void free();
    void printval(REAL, uint8_t ndigits) const;
    REAL get_val_from_global_index(len_t gi, len_t gj) const;
};



// -----------------------------------------------------------------------------
// public
// -----------------------------------------------------------------------------

// constructors/destructor

template <typename REAL>
mpimat<REAL>::mpimat(grid &blacs_grid)
{
  this->m = 0;
  this->n = 0;
  this->data = NULL;
  
  this->m_local = 0;
  this->n_local = 0;
  this->mb = 0;
  this->nb = 0;
  
  this->g = blacs_grid;
  
  this->free_data = true;
}



template <typename REAL>
mpimat<REAL>::mpimat(grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols)
{
  this->m_local = bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  this->n_local = bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  
  bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, this->m_local);
  
  size_t len = this->m_local * this->n_local * sizeof(REAL);
  this->data = (REAL*) std::malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
  
  this->m = nrows;
  this->n = ncols;
  this->mb = bf_rows;
  this->nb = bf_cols;
  this->g = blacs_grid;
  
  this->free_data = true;
}



template <typename REAL>
mpimat<REAL>::mpimat(grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct)
{
  this->m_local = bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  this->n_local = bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  
  bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, this->m_local);
  
  this->m = nrows;
  this->n = ncols;
  this->mb = bf_rows;
  this->nb = bf_cols;
  this->g = blacs_grid;
  
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
mpimat<REAL>::mpimat(const mpimat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  
  this->m_local = x.nrows_local();
  this->n_local = x.ncols_local();
  this->mb = x.bf_rows();
  this->nb = x.bf_cols();
  
  memcpy(this->desc, x.desc_ptr(), 9*sizeof(int));
  
  grid g = x.get_grid();
  this->g = g;
  
  this->data = x.data_ptr();
  
  this->free_data = x.should_free();
}



template <typename REAL>
mpimat<REAL>::~mpimat()
{
  this->free();
}



// memory management

template <typename REAL>
void mpimat<REAL>::resize(len_t nrows, len_t ncols, int bf_rows, int bf_cols)
{
  size_t len = (size_t) nrows * ncols * sizeof(REAL);
  size_t oldlen = (size_t) this->m * this->n * sizeof(REAL);
  
  if (len == oldlen && this->mb == bf_rows && this->nb == bf_cols)
  {
    this->m = nrows;
    this->n = ncols;
    return;
  }
  
  this->mb = bf_rows;
  this->nb = bf_cols;
  
  this->m_local = bcutils::numroc(nrows, this->mb, this->g.myrow(), 0, this->g.nprow());
  this->n_local = bcutils::numroc(ncols, this->nb, this->g.mycol(), 0, this->g.npcol());
  
  void *realloc_ptr = realloc(this->data, len);
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (REAL*) realloc_ptr;
  
  bcutils::descinit(this->desc, this->g.ictxt(), nrows, ncols, this->mb, this->nb, this->m_local);
  
  this->m = nrows;
  this->n = ncols;
}



template <typename REAL>
void mpimat<REAL>::set(grid &blacs_grid, REAL *data_, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct)
{
  this->free();
  
  m_local = bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  n_local = bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, m_local);
  
  this->m = nrows;
  this->n = ncols;
  this->mb = bf_rows;
  this->nb = bf_cols;
  this->g = blacs_grid;
  
  this->data = data_;
  
  this->free_data = free_on_destruct;
}



template <typename REAL>
mpimat<REAL> mpimat<REAL>::dupe() const
{
  mpimat<REAL> dup(this->g, this->m, this->n, this->mb, this->nb);
  
  size_t len = this->m_local * this->n_local * sizeof(REAL);
  memcpy(dup.data_ptr(), this->data, len);
  
  memcpy(dup.desc_ptr(), this->desc, 9*sizeof(int));
  
  return dup;
}



// printers

template <typename REAL>
void mpimat<REAL>::print(uint8_t ndigits) const
{
  for (len_t gi=0; gi<this->m; gi++)
  {
    for (len_t gj=0; gj<this->n; gj++)
    {
      int pr = bcutils::g2p(gi, this->mb, this->g.nprow());
      int pc = bcutils::g2p(gj, this->nb, this->g.npcol());
      
      int i = bcutils::g2l(gi, this->mb, this->g.nprow());
      int j = bcutils::g2l(gj, this->nb, this->g.npcol());
      
      REAL d;
      if (this->g.rank0())
      {
        if (pr == 0 && pc == 0)
          d = this->data[i + this->m_local*j];
        else
          this->g.recv(1, 1, &d, pr, pc);
        
        this->printval(d, ndigits);
      }
      else if (pr == this->g.myrow() && pc == this->g.mycol())
      {
        d = this->data[i + this->m_local*j];
        this->g.send(1, 1, &d, 0, 0);
      }
    }
    
    this->g.printf(0, 0, "\n");
  }
  
  this->g.printf(0, 0, "\n");
}



template <typename REAL>
void mpimat<REAL>::info() const
{
  if (this->g.rank0())
  {
    printf("# mpimat");
    printf(" %dx%d", this->m, this->n);
    printf(" on %dx%d grid", this->g.nprow(), this->g.npcol());
    printf(" type=%s", typeid(REAL).name());
    printf("\n");
  }
}



// fillers

template <typename REAL>
void mpimat<REAL>::fill_zero()
{
  size_t len = m_local * n_local * sizeof(REAL);
  memset(this->data, 0, len);
}



template <typename REAL>
void mpimat<REAL>::fill_one()
{
  this->fill_val((REAL) 1);
}



template <typename REAL>
void mpimat<REAL>::fill_val(const REAL v)
{
  #pragma omp parallel for simd
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
      this->data[i + this->m_local*j] = v;
  }
}



template <typename REAL>
void mpimat<REAL>::fill_linspace(REAL min, REAL max)
{
  if (min == max)
    this->fill_val(min);
  else
  {
    REAL v = (max-min)/((REAL) this->m*this->n - 1);
    for (len_t j=0; j<this->n_local; j++)
    {
      for (len_t i=0; i<this->m_local; i++)
      {
        int gi = bcutils::l2g(i, this->mb, this->g.nprow(), this->g.myrow());
        int gj = bcutils::l2g(j, this->nb, this->g.npcol(), this->g.mycol());
        
        this->data[i + this->m_local*j] = v*((REAL) gi + this->m*gj) + min;
      }
    }
  }
}



template <typename REAL>
void mpimat<REAL>::fill_eye()
{
  for (len_local_t j=0; j<n_local; j++)
  {
    for (len_local_t i=0; i<m_local; i++)
    {
      int gi = bcutils::l2g(i, this->mb, this->g.nprow(), this->g.myrow());
      int gj = bcutils::l2g(j, this->nb, this->g.npcol(), this->g.mycol());
      
      if (gi == gj)
        this->data[i + m_local*j] = 1;
      else
        this->data[i + m_local*j] = 0;
    }
  }
}



template <typename REAL>
void mpimat<REAL>::fill_runif(uint32_t seed, REAL min, REAL max)
{
  std::mt19937 mt(seed);
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
    {
      static std::uniform_real_distribution<REAL> dist(min, max);
      this->data[i + this->m_local*j] = dist(mt);
    }
  }
}

template <typename REAL>
void mpimat<REAL>::fill_runif(REAL min, REAL max)
{
  uint32_t seed = fmlutils::get_seed() + (g.myrow() + g.nprow()*g.mycol());
  this->fill_runif(seed, min, max);
}



template <typename REAL>
void mpimat<REAL>::fill_rnorm(uint32_t seed, REAL mean, REAL sd)
{
  std::mt19937 mt(seed);
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
    {
      static std::normal_distribution<REAL> dist(mean, sd);
      this->data[i + this->m_local*j] = dist(mt);
    }
  }
}

template <typename REAL>
void mpimat<REAL>::fill_rnorm(REAL mean, REAL sd)
{
  uint32_t seed = fmlutils::get_seed() + (g.myrow() + g.nprow()*g.mycol());
  this->fill_rnorm(seed, mean, sd);
}



template <typename REAL>
void mpimat<REAL>::scale(const REAL s)
{
  for (len_local_t j=0; j<this->n_local; j++)
  {
    for (len_local_t i=0; i<this->m_local; i++)
      this->data[i + this->m_local*j] *= s;
  }
}



// operators

template <typename REAL>
REAL mpimat<REAL>::operator()(len_t i)
{
  if (i < 0 || i >= (this->m * this->n))
    throw std::runtime_error("index out of bounds");
  
  int gi = i % this->m;
  int gj = i / this->m;
  
  REAL ret = this->get_val_from_global_index(gi, gj);
  return ret;
}

template <typename REAL>
const REAL mpimat<REAL>::operator()(len_t i) const
{
  if (i < 0 || i >= (this->m * this->n))
    throw std::runtime_error("index out of bounds");
  
  int gi = i % this->m;
  int gj = i / this->m;
  
  REAL ret = this->get_val_from_global_index(gi, gj);
  return ret;
}



template <typename REAL>
REAL mpimat<REAL>::operator()(len_t i, len_t j)
{
  if (i < 0 || i >= this->m || j < 0 || j >= this->n)
    throw std::runtime_error("index out of bounds");
  
  REAL ret = this->get_val_from_global_index(i, j);
  return ret;
}

template <typename REAL>
const REAL mpimat<REAL>::operator()(len_t i, len_t j) const
{
  if (i < 0 || i >= this->m || j < 0 || j >= this->n)
    throw std::runtime_error("index out of bounds");
  
  REAL ret = this->get_val_from_global_index(i, j);
  return ret;
}



template <typename REAL>
bool mpimat<REAL>::operator==(const mpimat<REAL> &x) const
{
  // same dim, same blocking, same grid
  if (this->m != x.nrows() || this->n != x.ncols())
    return false;
  
  if (this->mb != x.bf_rows() || this->nb != x.bf_cols())
    return false;
  
  if (this->g.ictxt() != x.g.ictxt())
    return false;
  
  const REAL *x_d = x.data_ptr();
  if (this->data == x_d)
    return true;
  
  int negation_ret = 0;
  for (len_t j=0; j<this->n_local; j++)
  {
    for (len_t i=0; i<this->m_local; i++)
    {
      if (this->data[i + this->m_local*j] != x_d[i + this->m_local*j])
      {
        negation_ret = 1;
        break;
      }
    }
  }
  
  g.allreduce(1, 1, &negation_ret, 'A');
  
  return !((bool) negation_ret);
}

template <typename REAL>
bool mpimat<REAL>::operator!=(const mpimat<REAL> &x) const
{
  return !(*this == x);
}



// -----------------------------------------------------------------------------
// private
// -----------------------------------------------------------------------------

template <typename REAL>
void mpimat<REAL>::free()
{
  if (this->free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <>
inline void mpimat<int>::printval(int val, uint8_t ndigits) const
{
  (void)ndigits;
  printf("%d ", val);
}

template <typename REAL>
void mpimat<REAL>::printval(REAL val, uint8_t ndigits) const
{
  printf("%.*f ", ndigits, val);
}



template <typename REAL>
REAL mpimat<REAL>::get_val_from_global_index(len_t gi, len_t gj) const
{
  REAL ret;
  
  int pr = bcutils::g2p(gi, this->mb, this->g.nprow());
  int pc = bcutils::g2p(gj, this->nb, this->g.npcol());
  
  int li = bcutils::g2l(gi, this->mb, this->g.nprow());
  int lj = bcutils::g2l(gj, this->nb, this->g.npcol());
  
  if (pr == this->g.myrow() && pc == this->g.mycol())
    ret = this->data[li + (this->m_local)*lj];
  else
    ret = (REAL) 0;
  
  g.allreduce(1, 1, &ret, 'A');
  
  return ret;
}


#endif
