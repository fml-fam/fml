// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_H
#define FML_MPI_LINALG_H
#pragma once


#include <stdexcept>

#include "../_internals/linalgutils.hh"
#include "../cpu/cpuvec.hh"

#include "internals/scalapack.hh"
#include "mpimat.hh"


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
    
    
    
    template <typename REAL>
    void check_grid(const mpimat<REAL> &a, const mpimat<REAL> &b)
    {
      if (a.get_grid().get_ictxt() != b.get_grid().get_ictxt())
        throw std::runtime_error("mpimat objects must be distributed on the same process grid");
    }
  }
  
  
  
  /**
    @brief Returns alpha*op(x) + beta*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha,beta Scalars.
    @param[in] x,y The inputs to the sum.
    @param[out] ret The sum.
    
    @except If x and y are inappropriately sized for the sum, the method will
    throw a 'runtime_error' exception.
    
    @impl Uses the PBLAS function `pXgeadd()`.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    check_grid(x, y);
    check_grid(x, ret);
    
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    scalapack::geadd(ctransy, m, n, beta, y.data_ptr(), y.desc_ptr(), (REAL) 0.0f, ret.data_ptr(), ret.desc_ptr());
    scalapack::geadd(ctransx, m, n, alpha, x.data_ptr(), x.desc_ptr(), (REAL) 1.0f, ret.data_ptr(), ret.desc_ptr());
  }
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    check_grid(x, y);
    
    len_t m, n;
    linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
  
  
  
  /**
    @brief Returns alpha*op(x)*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    
    @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. If the inputs are
     distributed on different grids, a `runtime_exception` is thrown.
    
    @impl Uses the PBLAS function `pXgemm()`.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  mpimat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    check_grid(x, y);
    
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
    @brief Computes ret = alpha*op(x)*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    @param[out] ret The product.
    
    @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. If the inputs are
     distributed on different grids, a `runtime_exception` is thrown.
    
    @impl Uses the PBLAS function `pXgemm()`.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    check_grid(x, y);
    check_grid(x, ret);
    
    int m, n, k;
    linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
  }
  
  
  
  /**
    @brief Computes lower triangle of alpha*x^T*x
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the BLAS function `pXsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void crossprod(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    check_grid(x, ret);
    
    const len_t n = x.ncols();
    
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    ret.fill_zero();
    scalapack::syrk('L', 'T', n, x.nrows(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> crossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    const len_t n = x.ncols();
    grid g = x.get_grid();
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    
    crossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
    @brief Computes lower triangle of alpha*x*x^T
    
    @param[in] alpha Scalar.
    @param[in] x Input data matrix.
    @param[out] ret The product.
    
    @impl Uses the PBLAS function `pXsyrk()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void tcrossprod(const REAL alpha, const mpimat<REAL> &x, mpimat<REAL> &ret)
  {
    check_grid(x, ret);
    
    const len_t m = x.nrows();
    
    if (m != ret.nrows() || m != ret.ncols())
      ret.resize(m, m);
    
    ret.fill_zero();
    scalapack::syrk('L', 'N', m, x.ncols(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
  }
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> tcrossprod(const REAL alpha, const mpimat<REAL> &x)
  {
    const len_t n = x.nrows();
    grid g = x.get_grid();
    mpimat<REAL> ret(g, n, n, x.bf_rows(), x.bf_cols());
    
    tcrossprod(alpha, x, ret);
    
    return ret;
  }
  
  
  
  /**
    @brief Computes the transpose out-of-place (i.e. in a copy).
    
    @param[in] x Input data matrix.
    @param[out] tx The transpose.
    
    @impl Uses the PBLAS function `pXtran()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void xpose(const mpimat<REAL> &x, mpimat<REAL> &tx)
  {
    check_grid(x, tx);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      tx.resize(n, m);
    
    scalapack::tran(n, m, 1.f, x.data_ptr(), x.desc_ptr(), 0.f, tx.data_ptr(), tx.desc_ptr());
  }
  
  /// \overload
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
  
  
  
  /**
    @brief Computes the PLU factorization with partial pivoting.
    
    @details The input is replaced by its LU factorization, with L
    unit-diagonal.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] p Vector of pivots, representing the diagonal matrix P in the
    PLU.
    
    @impl Uses the ScaLAPACK function `pXgetrf()`.
    
    @allocs If the pivot vector is inappropriately sized, it will automatically
    be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
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
  
  /// \overload
  template <typename REAL>
  int lu(mpimat<REAL> &x)
  {
    cpuvec<int> p;
    return lu(x, p);
  }
  
  
  
  /**
    @brief Computes the trace, i.e. the sum of the diagonal.
    
    @param[in] x Input data matrix.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL trace(const mpimat<REAL> &x)
  {
    const REAL *x_d = x.data_ptr();
    const len_t minmn = std::min(x.nrows(), x.ncols());
    const len_t m_local = x.nrows_local();
    const int mb = x.bf_rows();
    const int nb = x.bf_cols();
    grid g = x.get_grid();
    
    REAL tr = 0;
    for (len_t gi=0; gi<minmn; gi++)
    {
      const len_local_t i = bcutils::g2l(gi, mb, g.nprow());
      const len_local_t j = bcutils::g2l(gi, nb, g.npcol());
      
      const int pr = bcutils::g2p(gi, mb, g.nprow());
      const int pc = bcutils::g2p(gi, nb, g.npcol());
      
      if (pr == g.myrow() && pc == g.mycol())
        tr += x_d[i + m_local*j];
    }
    
    g.allreduce(1, 1, &tr, 'A');
    
    return tr;
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
  
  /**
    @brief Computes the singular value decomposition.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singnular vectors.
    
    @impl Uses the ScaLAPACK function `pXgesvd()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void svd(mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    mpimat<REAL> ignored(x.get_grid());
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    check_info(info, "gesvd");
  }
  
  /// \overload
  template <typename REAL>
  void svd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    check_grid(x, u);
    check_grid(x, vt);
    
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
  
  /**
    @brief Compute the eigenvalues and optionally the eigenvectors for a
    symmetric matrix.
    
    @details The input data is overwritten.
    
    @param[inout] x Input data matrix. Should be square.
    @param[out] values Eigenvalues.
    @param[out] vectors Eigenvectors.
    
    @impl Uses the ScaLAPACK functions `pXsyevr()`.
    
    @allocs If any output's dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void eigen_sym(mpimat<REAL> &x, cpuvec<REAL> &values)
  {
    mpimat<REAL> ignored(x.get_grid());
    
    int info = eig_sym_internals(true, x, values, ignored);
    check_info(info, "syevr");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(mpimat<REAL> &x, cpuvec<REAL> &values, mpimat<REAL> &vectors)
  {
    check_grid(x, vectors);
    
    int info = eig_sym_internals(false, x, values, vectors);
    check_info(info, "syevr");
  }
  
  
  
  /**
    @brief Compute the matrix inverse.
    
    @details The input is replaced by its inverse, computed via a PLU.
    
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the ScaLAPACK functions `pXgetrf()` (LU) and `pXgetri()`
    (solve).
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
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
  
  
  
  /**
    @brief Solve a system of equations.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input LHS. Should be square. Overwritten by LU.
    @param[inout] y Input RHS. Overwritten by solution.
    
    @impl Uses the LAPACK functions `pXgesv()`.
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square or if the RHS is incompatible with the
    LHS, a `runtime_error` exception is thrown. If the inputs are distributed
    on different grids, a `runtime_exception` is thrown. If an allocation
    fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void solve(mpimat<REAL> &x, mpimat<REAL> &y)
  {
    check_grid(x, y);
    
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
