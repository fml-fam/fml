#ifndef FML_MPI_LINALG_H
#define FML_MPI_LINALG_H


#include <stdexcept>

#include "../linalgutils.hh"
#include "../cpu/cpuvec.hh"

#include "mpimat.hh"
#include "scalapack.hh"


namespace linalg
{
  namespace
  {
    inline void check_info(const int info, std::string fun)
    {
      if (info != 0)
      {
        std::string msg = "ScaLAPACK function " + fun + "() returned info=" + std::to_string(info);
        throw std::runtime_error(msg);
      }
    }
  }
  
  
  
  // ret = alpha*op(x) + beta*op(y)
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      throw std::runtime_error("non-conformable arguments");
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    scalapack::geadd(ctransy, m, n, beta, y.data_ptr(), y.desc_ptr(), (REAL) 0.0f, ret.data_ptr(), ret.desc_ptr());
    scalapack::geadd(ctransx, m, n, alpha, x.data_ptr(), x.desc_ptr(), (REAL) 1.0f, ret.data_ptr(), ret.desc_ptr());
  }
  
  template <typename REAL>
  mpimat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
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
   * @impl Uses the PBLAS function pXgemm().
   * 
   * @comm The method will communicate across all processes in the BLACS grid.
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  mpimat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
    
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
   * @impl Uses the PBLAS function pXgemm().
   * 
   * @comm The method will communicate across all processes in the BLACS grid.
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    int m, n, k;
    
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
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
   * @impl Uses the BLAS function pXsyrk().
   * 
   * @comm The method will communicate across all processes in the BLACS grid.
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    scalapack::syrk('L', 'T', n, x.nrows(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  /**
   * @brief Returns lower triangle of alpha*x^T*x
   * 
   * @param[in] alpha Scalar.
   * @param[in] x Input data matrix.
   * 
   * @impl Uses the BLAS function pXsyrk().
   * 
   * @comm The method will communicate across all processes in the BLACS grid.
   * 
   * @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  mpimat<REAL> crossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    const len_t n = x.ncols();
    grid g = x.get_grid();
    
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    ret.fill_zero();
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void tcrossprod(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    const len_t m = x.nrows();
    
    if (m != ret.nrows() || m != ret.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    scalapack::syrk('L', 'N', m, x.ncols(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  template <typename REAL>
  mpimat<REAL> tcrossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    const len_t n = x.nrows();
    grid g = x.get_grid();
    
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    ret.fill_zero();
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  template <typename REAL>
  void xpose(const mpimat<REAL> &x, mpimat<REAL> &tx)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    scalapack::tran(n, m, 1.f, x.data_ptr(), x.desc_ptr(), 0.f, tx.data_ptr(), tx.desc_ptr());
  }
  
  template <typename REAL>
  mpimat<REAL> xpose(const mpimat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    grid g = x.get_grid();
    
    mpimat<REAL> tx(g, n, m, x.bf_rows(), x.bf_cols());
    xpose(x, tx);
    return tx;
  }
  
  
  
  template <typename REAL>
  int lu(mpimat<REAL> &x, cpuvec<int> &p)
  {
    int info = 0;
    const len_t m = x.nrows();
    const len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    scalapack::getrf(m, x.ncols(), x.data_ptr(), x.desc_ptr(), p.data_ptr(), &info);
    
    return info;
  }
  
  template <typename REAL>
  int lu(mpimat<REAL> &x)
  {
    cpuvec<int> p;
    return lu(x, p);
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int svd_internals(const int nu, const int nv, mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
    {
      int info = 0;
      char jobu, jobvt;
      
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      s.resize(minmn);
      
      if (nu == 0 && nv == 0)
      {
        jobu = 'N';
        jobvt = 'N';
      }
      else if (nu <= minmn && nv <= minmn)
      {
        jobu = 'V';
        jobvt = 'V';
        
        const int mb = x.bf_rows();
        const int nb = x.bf_cols();
        
        u.resize(m, minmn, mb, nb);
        vt.resize(minmn, n, mb, nb);
      }
      else
      {
        // TODO
      }
      
      REAL tmp;
      scalapack::gesvd(jobu, jobvt, m, n, x.data_ptr(), x.desc_ptr(), s.data_ptr(), u.data_ptr(), u.desc_ptr(), vt.data_ptr(), vt.desc_ptr(), &tmp, -1, &info);
      int lwork = (int) tmp;
      cpuvec<REAL> work(lwork);
      
      scalapack::gesvd(jobu, jobvt, m, n, x.data_ptr(), x.desc_ptr(), s.data_ptr(), u.data_ptr(), u.desc_ptr(), vt.data_ptr(), vt.desc_ptr(), work.data_ptr(), lwork, &info);
      
      return info;
    }
  }
  
  template <typename REAL>
  void svd(mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    mpimat<REAL> ignored;
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    check_info(info, "gesvd");
  }
  
  template <typename REAL>
  void svd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    int info = svd_internals(1, 1, x, s, u, vt);
    check_info(info, "gesvd");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, mpimat<REAL> &x,
      cpuvec<REAL> &values, mpimat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      int info = 0;
      int val_found, vec_found;
      char jobz;
      
      len_t n = x.nrows();
      values.resize(n);
      
      if (only_values)
        jobz = 'N';
      else
      {
        jobz = 'V';
        vectors.resize(n, n, x.bf_rows(), x.bf_cols());
      }
      
      REAL worksize;
      int lwork, liwork;
      
      scalapack::syevr(jobz, 'A', 'U', n, x.data_ptr(), x.desc_ptr(),
        (REAL) 0.f, (REAL) 0.f, 0, 0, &val_found, &vec_found,
        values.data_ptr(), vectors.data_ptr(), vectors.desc_ptr(),
        &worksize, -1, &liwork, -1, &info);
      
      lwork = (int) worksize;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      scalapack::syevr(jobz, 'A', 'U', n, x.data_ptr(), x.desc_ptr(),
        (REAL) 0.f, (REAL) 0.f, 0, 0, &val_found, &vec_found,
        values.data_ptr(), vectors.data_ptr(), vectors.desc_ptr(),
        work.data_ptr(), lwork, iwork.data_ptr(), liwork, &info);
      
      return info;
    }
  }
  
  template <typename REAL>
  void eigen(bool symmetric, mpimat<REAL> &x, cpuvec<REAL> &values)
  {
    mpimat<REAL> ignored;
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
  void eigen(bool symmetric, mpimat<REAL> &x, cpuvec<REAL> &values,
    mpimat<REAL> &vectors)
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
  void invert(mpimat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    cpuvec<int> p;
    int info = lu(x, p);
    check_info(info, "getrf");
    
    // Invert
    const len_t n = x.nrows();
    REAL tmp;
    int liwork;
    scalapack::getri(n, x.data_ptr(), x.desc_ptr(), p.data_ptr(), &tmp, -1, &liwork, -1, &info);
    int lwork = (int) tmp;
    cpuvec<REAL> work(lwork);
    cpuvec<int> iwork(liwork);
    
    scalapack::getri(n, x.data_ptr(), x.desc_ptr(), p.data_ptr(), work.data_ptr(), lwork, iwork.data_ptr(), liwork, &info);
    check_info(info, "getri");
  }
  
  
  
  template <typename REAL>
  void solve(mpimat<REAL> &x, mpimat<REAL> &y)
  {
    const len_t n = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    if (n != y.nrows())
      throw std::runtime_error("rhs 'y' must be compatible with data matrix 'x'");
    
    int info;
    cpuvec<int> p(n);
    scalapack::gesv(n, y.ncols(), x.data_ptr(), x.desc_ptr(), p.data_ptr(), y.data_ptr(), y.desc_ptr(), &info);
    check_info(info, "gesv");
  }
}


#endif
