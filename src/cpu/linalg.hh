#ifndef FML_CPU_LINALG_H
#define FML_CPU_LINALG_H


#include <cmath>
#include <stdexcept>

#include "../linalgutils.hh"
#include "../omputils.hh"

#include "cpumat.hh"
#include "cpuvec.hh"
#include "lapack.hh"


/**
 * @brief Linear algebra functions.
*/
namespace linalg
{
  namespace
  {
    inline void check_info(const int info, std::string fun)
    {
      if (info != 0)
      {
        std::string msg = "LAPACK function " + fun + "() returned info=" + std::to_string(info);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  // ret = alpha*op(x) + beta*op(y)
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const cpumat<REAL> &x, const cpumat<REAL> &y, cpumat<REAL> &ret)
  {
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      throw std::runtime_error("non-conformable arguments");
    
    const REAL *x_d = x.data_ptr();
    const REAL *y_d = y.data_ptr();
    REAL *ret_d = ret.data_ptr();
    
    if (!transx && !transy)
    {
      #pragma omp parallel for if(m*n > omputils::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          ret_d[i + m*j] = alpha*x_d[i + m*j] + beta*y_d[i + m*j];
      }
    }
    else if (transx && transy)
    {
      #pragma omp parallel for if(m*n > omputils::OMP_MIN_SIZE)
      for (len_t j=0; j<m; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<n; i++)
          ret_d[j + m*i] = alpha*x_d[i + n*j] + beta*y_d[i + n*j];
      }
    }
    else if (transx && !transy)
    {
      #pragma omp parallel for if(m*n > omputils::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          ret_d[i + m*j] = alpha*x_d[j + n*i] + beta*y_d[i + m*j];
      }
    }
    else if (!transx && transy)
    {
      #pragma omp parallel for if(m*n > omputils::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          ret_d[i + m*j] = alpha*x_d[i + m*j] + beta*y_d[j + n*i];
      }
    }
  }
  
  template <typename REAL>
  cpumat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    cpumat<REAL> ret(m, n);
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
  
  
  
  /**
   * @brief Returns alpha*op(x)*op(y) where op(A) is A or A^T
   * 
   * @param[in] transx Should x^T be used?
   * @param[in] transy Should y^T be used?
   * @param[in] alpha Scalar.
   * @param[in] x Left multiplicand.
   * @param[in] y Right multiplicand.
   * 
   * @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. Likewise for ret.
   * 
   * @impl Uses the BLAS function Xgemm().
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  cpumat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const cpumat<REAL> &x, const cpumat<REAL> &y)
  {
    int m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    cpumat<REAL> ret(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
    
    return ret;
  }
  
  /**
   * @brief Computes ret = alpha*op(x)*op(y) where op(A) is A or A^T
   * 
   * @param[in] transx Should x^T be used?
   * @param[in] transy Should y^T be used?
   * @param[in] alpha Scalar.
   * @param[in] x Left multiplicand.
   * @param[in] y Right multiplicand.
   * @param[out] ret The product.
   * 
   * @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. Likewise for ret.
   * 
   * @impl Uses the BLAS function Xgemm().
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const cpumat<REAL> &x, const cpumat<REAL> &y, cpumat<REAL> &ret)
  {
    int m, n, k;
    const len_t mx = x.nrows();
    const len_t my = y.nrows();
    
    linalgutils::matmult_params(transx, transy, mx, x.ncols(), my, y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    lapack::gemm(ctransx, ctransy, m, n, k, alpha, x.data_ptr(), mx, y.data_ptr(), my, (REAL)0, ret.data_ptr(), m);
  }
  
  
  
  /**
   * @brief Computes lower triangle of alpha*x^T*x
   * 
   * @param[in] alpha Scalar.
   * @param[in] x Input data matrix.
   * @param[out] ret The product.
   * 
   * @except If ret is inappropriately sized for the product, the method will
     throw a 'runtime_error' exception.
   * 
   * @impl Uses the BLAS function Xsyrk().
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const cpumat<REAL> &x, cpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    lapack::syrk('L', 'T', n, m, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), n);
  }
  
  /**
   * @brief Returns lower triangle of alpha*x^T*x
   * 
   * @param[in] alpha Scalar.
   * @param[in] x Input data matrix.
   * 
   * @impl Uses the BLAS function Xsyrk().
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  cpumat<REAL> crossprod(const REAL alpha, const cpumat<REAL> &x)
  {
    const len_t n = x.ncols();
    
    cpumat<REAL> ret(n, n);
    ret.fill_zero();
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void tcrossprod(const REAL alpha, const cpumat<REAL> &x, cpumat<REAL> &ret)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != ret.nrows() || m != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    lapack::syrk('L', 'N', m, n, alpha, x.data_ptr(), m, (REAL)0.0, ret.data_ptr(), m);
  }
  
  template <typename REAL>
  cpumat<REAL> tcrossprod(const REAL alpha, const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    
    cpumat<REAL> ret(m, m);
    ret.fill_zero();
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void xpose(const cpumat<REAL> &x, cpumat<REAL> &tx)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    const int blocksize = 8;
    const REAL *x_d = x.data_ptr();
    REAL *tx_d = tx.data_ptr();
    
    #pragma omp parallel for shared(tx) schedule(dynamic, 1) if(m*n>omputils::OMP_MIN_SIZE)
    for (int j=0; j<n; j+=blocksize)
    {
      for (int i=0; i<m; i+=blocksize)
      {
        for (int col=j; col<j+blocksize && col<n; ++col)
        {
          for (int row=i; row<i+blocksize && row<m; ++row)
            tx_d[col + n*row] = x_d[row + m*col];
        }
      }
    }
  }
  
  template <typename REAL>
  cpumat<REAL> xpose(const cpumat<REAL> &x)
  {
    cpumat<REAL> tx(x.ncols(), x.nrows());
    xpose(x, tx);
    return tx;
  }
  
  
  
  template <typename REAL>
  int lu(cpumat<REAL> &x, cpuvec<int> &p)
  {
    int info = 0;
    const len_t m = x.nrows();
    const len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    lapack::getrf(m, x.ncols(), x.data_ptr(), m, p.data_ptr(), &info);
    
    return info;
  }
  
  template <typename REAL>
  int lu(cpumat<REAL> &x)
  {
    cpuvec<int> p;
    return lu(x, p);
  }
  
  
  
  template <typename REAL>
  void det(cpumat<REAL> &x, int &sign, REAL &modulus)
  {
    const len_t m = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    cpuvec<int> p;
    int info = lu(x, p);
    
    if (info != 0)
    {
      if (info > 0)
      {
        sign = 1;
        modulus = -INFINITY;
        return;
      }
      else
        return;
    }
    
    
    // get determinant
    REAL mod = 0.0;
    int sgn = 1;
    
    const int *ipiv = p.data_ptr();
    for (int i=0; i<m; i++)
    {
      if (ipiv[i] != (i + 1))
        sgn = -sgn;
    }
    
    const REAL *a = x.data_ptr();
    
    #pragma omp parallel for reduction(+:mod) reduction(*:sgn)
    for (int i=0; i<m; i+=m+1)
    {
      const REAL d = a[i + m*i];
      if (d < 0)
      {
        mod += log(-d);
        sgn *= -1;
      }
      else
        mod += log(d);
    }
    
    modulus = mod;
    sign = sgn;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int svd_internals(const int nu, const int nv, cpumat<REAL> &x, cpuvec<REAL> &s,
      cpumat<REAL> &u, cpumat<REAL> &vt)
    {
      int info = 0;
      char jobz;
      int ldvt;
      
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      s.resize(minmn);
      
      if (nu == 0 && nv == 0)
      {
        jobz = 'N';
        ldvt = 1; // value is irrelevant, but must exist!
      }
      else if (nu <= minmn && nv <= minmn)
      {
        jobz = 'S';
        ldvt = minmn;
        
        u.resize(m, minmn);
        vt.resize(minmn, n);
      }
      else
      {
        jobz = 'A';
        ldvt = n;
      }
      
      cpuvec<int> iwork(8*minmn);
      
      REAL tmp;
      lapack::gesdd(jobz, m, n, x.data_ptr(), m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), ldvt, &tmp, -1, iwork.data_ptr(), &info);
      int lwork = (int) tmp;
      cpuvec<REAL> work(lwork);
      
      lapack::gesdd(jobz, m, n, x.data_ptr(), m, s.data_ptr(), u.data_ptr(), m, vt.data_ptr(), ldvt, work.data_ptr(), lwork, iwork.data_ptr(), &info);
      
      return info;
    }
  }
  
  template <typename REAL>
  void svd(cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    cpumat<REAL> ignored;
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    check_info(info, "gesdd");
  }
  
  template <typename REAL>
  void svd(cpumat<REAL> &x, cpuvec<REAL> &s, cpumat<REAL> &u, cpumat<REAL> &vt)
  {
    int info = svd_internals(1, 1, x, s, u, vt);
    check_info(info, "gesdd");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, cpumat<REAL> &x,
      cpuvec<REAL> &values, cpumat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      int info = 0;
      int nfound;
      char jobz;
      
      len_t n = x.nrows();
      values.resize(n);
      cpuvec<int> support;
      
      if (only_values)
        jobz = 'N';
      else
      {
        jobz = 'V';
        vectors.resize(n, n);
        support.resize(2*n);
      }
      
      REAL worksize;
      int lwork, liwork;
      lapack::syevr(jobz, 'A', 'U', n, x.data_ptr(), n, (REAL) 0.f, (REAL) 0.f,
        0, 0, (REAL) 0.f, &nfound, values.data_ptr(), vectors.data_ptr(), n,
        support.data_ptr(), &worksize, -1, &liwork, -1,
        &info);
      
      lwork = (int) worksize;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      lapack::syevr(jobz, 'A', 'U', n, x.data_ptr(), n, (REAL) 0.f, (REAL) 0.f,
        0, 0, (REAL) 0.f, &nfound, values.data_ptr(), vectors.data_ptr(), n,
        support.data_ptr(), work.data_ptr(), lwork, iwork.data_ptr(), liwork,
        &info);
      
      values.rev();
      
      if (!only_values)
        vectors.rev_cols();
      
      return info;
    }
  }
  
  template <typename REAL>
  void eigen(bool symmetric, cpumat<REAL> &x, cpuvec<REAL> &values)
  {
    cpumat<REAL> ignored;
    if (symmetric)
    {
      int info = eig_sym_internals(true, x, values, ignored);
      check_info(info, "syevr");
    }
    else
    {
      // TODO
    }
  }
  
  template <typename REAL>
  void eigen(bool symmetric, cpumat<REAL> &x, cpuvec<REAL> &values,
    cpumat<REAL> &vectors)
  {
    if (symmetric)
    {
      int info = eig_sym_internals(false, x, values, vectors);
      check_info(info, "syevr");
    }
    else
    {
      // TODO
    }
  }
  
  
  
  template <typename REAL>
  void invert(cpumat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    cpuvec<int> p;
    int info = lu(x, p);
    check_info(info, "getrf");
    
    // Invert
    REAL tmp;
    lapack::getri(n, x.data_ptr(), n, p.data_ptr(), &tmp, -1, &info);
    int lwork = (int) tmp;
    cpuvec<REAL> work(lwork);
    
    lapack::getri(n, x.data_ptr(), n, p.data_ptr(), work.data_ptr(), lwork, &info);
    check_info(info, "getri");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void solver(cpumat<REAL> &x, len_t ylen, len_t nrhs, REAL *y_d)
    {
      const len_t n = x.nrows();
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      if (n != ylen)
        throw std::runtime_error("rhs 'y' must be compatible with data matrix 'x'");
      
      int info;
      cpuvec<int> p(n);
      lapack::gesv(n, nrhs, x.data_ptr(), n, p.data_ptr(), y_d, n, &info);
      check_info(info, "gesv");
    }
  }
  
  template <typename REAL>
  void solve(cpumat<REAL> &x, cpuvec<REAL> &y)
  {
    solver(x, y.size(), 1, y.data_ptr());
  }
  
  template <typename REAL>
  void solve(cpumat<REAL> &x, cpumat<REAL> &y)
  {
    solver(x, y.nrows(), y.ncols(), y.data_ptr());
  }
}


#endif
