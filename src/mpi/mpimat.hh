#ifndef FML_MPIMAT_H
#define FML_MPIMAT_H


#include <cstdlib>
#include <cstring>
#include <random>

#include "grid.hh"
#include "bcutils.hh"
#include "../matrix.hh"

typedef int len_local_t;


template <typename REAL>
class mpimat : public matrix<REAL>
{
  public:
    mpimat();
    mpimat(grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows=16, int bf_cols=16);
    mpimat(REAL *data_, grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct=false);
    mpimat(const mpimat &x);
    ~mpimat();
    
    void resize(len_t nrows, len_t ncols);
    void set(REAL *data_, grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct=false);
    mpimat<REAL> dupe();
    
    void print(uint8_t ndigits=4);
    void info();
    
    void fill_zero();
    void fill_eye();
    void fill_runif(int seed, REAL min=0, REAL max=1);
    
    void scale(const REAL s);
    
    len_local_t nrows_local() const {return m_local;};
    len_local_t ncols_local() const {return n_local;};
    int bf_rows() const {return mb;};
    int bf_cols() const {return nb;};
    int* desc_ptr() {return desc;};
    const int* desc_ptr() const {return desc;};
    grid get_grid() const {return g;};
    
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
    void printval(uint8_t ndigits, len_t i, len_t j);
};



template <typename REAL>
mpimat<REAL>::mpimat()
{
  this->data = NULL;
  
  this->free_data = true;
}



template <typename REAL>
mpimat<REAL>::mpimat(grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols)
{
  m_local = bcutils::numroc(nrows, bf_rows, blacs_grid.myrow(), 0, blacs_grid.nprow());
  n_local = bcutils::numroc(ncols, bf_cols, blacs_grid.mycol(), 0, blacs_grid.npcol());
  bcutils::descinit(this->desc, blacs_grid.ictxt(), nrows, ncols, bf_rows, bf_cols, m_local);
  
  size_t len = m_local * n_local * sizeof(REAL);
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
mpimat<REAL>::mpimat(REAL *data_, grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct)
{
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
mpimat<REAL>::mpimat(const mpimat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  
  this->m_local = x.nrows_local();
  this->n_local = x.ncols_local();
  this->mb = x.bf_rows();
  this->nb = x.bf_cols();
  
  const int *xdesc = x.desc_ptr();
  for (int i=0; i<9; i++)
    this->desc[i] = xdesc[i];
  
  grid g = x.get_grid();
  this->g = g;
  
  this->data = x.data_ptr();
  
  this->free_data = x.should_free();
}



template <typename REAL>
mpimat<REAL>::~mpimat()
{
  if (free_data && this->data)
  {
    std::free(this->data);
    this->data = NULL;
  }
}



template <typename REAL>
void mpimat<REAL>::resize(len_t nrows, len_t ncols)
{
  m_local = bcutils::numroc(nrows, this->mb, this->g.myrow(), 0, this->g.nprow());
  n_local = bcutils::numroc(ncols, this->nb, this->g.mycol(), 0, this->g.npcol());
  
  size_t len = m_local * n_local * sizeof(REAL);
  void *realloc_ptr = realloc(this->data, len);
  if (realloc_ptr == NULL)
    throw std::bad_alloc();
  
  this->data = (REAL*) realloc_ptr;
  
  bcutils::descinit(this->desc, this->g.ictxt(), nrows, ncols, this->mb, this->nb, m_local);
  
  this->m = nrows;
  this->n = ncols;
}



template <typename REAL>
void mpimat<REAL>::set(REAL *data_, grid &blacs_grid, len_t nrows, len_t ncols, int bf_rows, int bf_cols, bool free_on_destruct)
{
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
mpimat<REAL> mpimat<REAL>::dupe()
{
  mpimat<REAL> dup(this->g, this->m, this->n, this->mb, this->nb);
  
  size_t len = this->m_local * this->n_local * sizeof(REAL);
  memcpy(dup.data_ptr(), this->data, len);
  
  return dup;
}



template <>
void mpimat<int>::printval(uint8_t ndigits, len_t i, len_t j)
{
  (void)ndigits;
  printf("%d ", this->data[i + this->m*j]);
}

template <typename REAL>
void mpimat<REAL>::printval(uint8_t ndigits, len_t i, len_t j)
{
  printf("%.*f ", ndigits, this->data[i + this->m*j]);
}

template <typename REAL>
void mpimat<REAL>::print(uint8_t ndigits)
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
          d = this->data[i + this->m*j];
        else
          this->g.recv(1, 1, &d, pr, pc);
        
        this->printval(ndigits, i, j);
      }
      else if (pr == this->g.myrow() && pc == this->g.mycol())
      {
        d = this->data[i + this->m*j];
        this->g.send(1, 1, &d, 0, 0);
      }
    }
    
    this->g.printf(0, 0, "\n");
  }
  
  this->g.printf(0, 0, "\n");
}



template <typename REAL>
void mpimat<REAL>::info()
{
  printf("# mpimat");
  printf(" %dx%d", this->m, this->n);
  printf(" on %dx%d grid", this->g.nprow(), this->g.npcol());
  printf(" type=%s", typeid(REAL).name());
  printf("\n");
}



template <typename REAL>
void mpimat<REAL>::fill_zero()
{
  size_t len = m_local * n_local * sizeof(REAL);
  memset(this->data, 0, len);
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
void mpimat<REAL>::fill_runif(int seed, REAL min, REAL max)
{
  
}



template <typename REAL>
void mpimat<REAL>::scale(const REAL s)
{
  for (len_local_t j=0; j<this->m_local; j++)
  {
    for (len_local_t i=0; i<this->n_local; i++)
      this->data[i + this->m_local*j] *= s;
  }
}


#endif
