#ifndef FML_CPUMAT_H
#define FML_CPUMAT_H


#include <cstdlib>
#include <cstring>
#include <random>

#include "../matrix.hh"


template <typename REAL>
class cpumat : public matrix<REAL>
{
  public:
    cpumat(){};
    cpumat(len_t nrows, len_t ncols);
    cpumat(REAL *data, len_t nrows, len_t ncols);
    cpumat(const cpumat &x);
    
    void free();
    
    void print(uint8_t ndigits=4);
    
    void fill_zero();
    void fill_eye();
    void fill_runif(int seed, REAL min=0, REAL max=1);
    
    void scale(const REAL s);
};



template <typename REAL>
cpumat<REAL>::cpumat(len_t nrows, len_t ncols)
{
  size_t len = nrows * ncols * sizeof(REAL);
  this->data = (REAL*) malloc(len);
  if (this->data == NULL)
    throw std::bad_alloc();
  
  this->m = nrows;
  this->n = ncols;
}



template <typename REAL>
cpumat<REAL>::cpumat(REAL *data_, len_t nrows, len_t ncols)
{
  this->m = nrows;
  this->n = ncols;
  this->data = data_;
}



template <typename REAL>
cpumat<REAL>::cpumat(const cpumat<REAL> &x)
{
  this->m = x.nrows();
  this->n = x.ncols();
  this->data = x.data_ptr();
}



template <typename REAL>
void cpumat<REAL>::free()
{
  if (this->data)
  {
    free(this->data);
    this->data = NULL;
  }
}



template <>
void cpumat<int>::printval(uint8_t ndigits, len_t i, len_t j)
{
  (void)ndigits;
  printf("%d ", this->data[i + this->m*j]);
}

template <>
void cpumat<float>::printval(uint8_t ndigits, len_t i, len_t j)
{
  printf("%.*f ", ndigits, this->data[i + this->m*j]);
}

template <typename REAL>
void cpumat<REAL>::print(uint8_t ndigits)
{
  for (len_t i=0; i<this->m; i++)
  {
    for (len_t j=0; j<this->n; j++)
      this->printval(ndigits, i, j);
    
    putchar('\n');
  }
  
  putchar('\n');
}



template <typename REAL>
void cpumat<REAL>::fill_zero()
{
  size_t len = (this->m) * (this->n) * sizeof(REAL);
  memset(this->data, 0, len);
}



template <typename REAL>
void cpumat<REAL>::fill_eye()
{
  this->fill_zero();
  for (len_t i=0; i<this->m && i<this->n; i++)
    this->data[i + this->m*i] = 1;
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
void cpumat<REAL>::scale(const REAL s)
{
  for (len_t j=0; j<this->m; j++)
  {
    for (len_t i=0; i<this->n; i++)
      this->data[i + this->m*j] *= s;
  }
}


#endif
