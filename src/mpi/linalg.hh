// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_H
#define FML_MPI_LINALG_H
#pragma once


#include <stdexcept>

#include "../_internals/linalgutils.hh"
#include "../cpu/cpuvec.hh"

#include "internals/bcutils.hh"
#include "internals/scalapack.hh"
#include "mpimat.hh"
#include "mpihelpers.hh"


namespace linalg
{
  namespace
  {
    template <typename REAL>
    void check_grid(const mpimat<REAL> &a, const mpimat<REAL> &b)
    {
      if (a.get_grid().ictxt() != b.get_grid().ictxt())
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
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @impl Uses the PBLAS function `pXgeadd()`.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    check_grid(x, y);
    check_grid(x, ret);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    fml::scalapack::geadd(ctransy, m, n, beta, y.data_ptr(), y.desc_ptr(), (REAL) 0.0f, ret.data_ptr(), ret.desc_ptr());
    fml::scalapack::geadd(ctransx, m, n, alpha, x.data_ptr(), x.desc_ptr(), (REAL) 1.0f, ret.data_ptr(), ret.desc_ptr());
  }
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    check_grid(x, y);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
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
    fml::linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
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
    fml::linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::scalapack::gemm(ctransx, ctransy, m, n, k, alpha,
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
    fml::scalapack::syrk('L', 'T', n, x.nrows(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
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
    fml::scalapack::syrk('L', 'N', m, x.ncols(), alpha, x.data_ptr(), x.desc_ptr(), (REAL) 0, ret.data_ptr(), ret.desc_ptr());
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
    
    @comm The method will communicate across all processes in the BLACS grid.
    
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
    
    fml::scalapack::tran(n, m, 1.f, x.data_ptr(), x.desc_ptr(), 0.f, tx.data_ptr(), tx.desc_ptr());
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
    @param[out] info The ScaLAPACK return number.
    
    @impl Uses the ScaLAPACK function `pXgetrf()`.
    
    @allocs If the pivot vector is inappropriately sized, it will automatically
    be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lu(mpimat<REAL> &x, cpuvec<int> &p, int &info)
  {
    info = 0;
    const len_t m = x.nrows();
    const len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    fml::scalapack::getrf(m, x.ncols(), x.data_ptr(), x.desc_ptr(), p.data_ptr(), &info);
  }
  
  /// \overload
  template <typename REAL>
  void lu(mpimat<REAL> &x)
  {
    cpuvec<int> p;
    int info;
    
    lu(x, p, info);
    
    fml::linalgutils::check_info(info, "getrf");
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
      const len_local_t i = fml::bcutils::g2l(gi, mb, g.nprow());
      const len_local_t j = fml::bcutils::g2l(gi, nb, g.npcol());
      
      const int pr = fml::bcutils::g2p(gi, mb, g.nprow());
      const int pc = fml::bcutils::g2p(gi, nb, g.npcol());
      
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
      fml::scalapack::gesvd(jobu, jobvt, m, n, x.data_ptr(), x.desc_ptr(), s.data_ptr(), u.data_ptr(), u.desc_ptr(), vt.data_ptr(), vt.desc_ptr(), &tmp, -1, &info);
      int lwork = (int) tmp;
      cpuvec<REAL> work(lwork);
      
      fml::scalapack::gesvd(jobu, jobvt, m, n, x.data_ptr(), x.desc_ptr(), s.data_ptr(), u.data_ptr(), u.desc_ptr(), vt.data_ptr(), vt.desc_ptr(), work.data_ptr(), lwork, &info);
      
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
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void svd(mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    mpimat<REAL> ignored(x.get_grid());
    int info = svd_internals(0, 0, x, s, ignored, ignored);
    fml::linalgutils::check_info(info, "gesvd");
  }
  
  /// \overload
  template <typename REAL>
  void svd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    check_grid(x, u);
    check_grid(x, vt);
    
    int info = svd_internals(1, 1, x, s, u, vt);
    fml::linalgutils::check_info(info, "gesvd");
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
      
      fml::scalapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), x.desc_ptr(),
        (REAL) 0.f, (REAL) 0.f, 0, 0, &val_found, &vec_found,
        values.data_ptr(), vectors.data_ptr(), vectors.desc_ptr(),
        &worksize, -1, &liwork, -1, &info);
      
      lwork = (int) worksize;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      fml::scalapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), x.desc_ptr(),
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
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void eigen_sym(mpimat<REAL> &x, cpuvec<REAL> &values)
  {
    mpimat<REAL> ignored(x.get_grid());
    
    int info = eig_sym_internals(true, x, values, ignored);
    fml::linalgutils::check_info(info, "syevr");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(mpimat<REAL> &x, cpuvec<REAL> &values, mpimat<REAL> &vectors)
  {
    check_grid(x, vectors);
    
    int info = eig_sym_internals(false, x, values, vectors);
    fml::linalgutils::check_info(info, "syevr");
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
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void invert(mpimat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    cpuvec<int> p;
    int info;
    lu(x, p, info);
    fml::linalgutils::check_info(info, "getrf");
    
    // Invert
    const len_t n = x.nrows();
    REAL tmp;
    int liwork;
    fml::scalapack::getri(n, x.data_ptr(), x.desc_ptr(), p.data_ptr(), &tmp, -1, &liwork, -1, &info);
    int lwork = (int) tmp;
    cpuvec<REAL> work(lwork);
    cpuvec<int> iwork(liwork);
    
    fml::scalapack::getri(n, x.data_ptr(), x.desc_ptr(), p.data_ptr(), work.data_ptr(), lwork, iwork.data_ptr(), liwork, &info);
    fml::linalgutils::check_info(info, "getri");
  }
  
  
  
  /**
    @brief Solve a system of equations.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input LHS. Should be square. Overwritten by LU.
    @param[inout] y Input RHS. Overwritten by solution.
    
    @impl Uses the ScaLAPACK functions `pXgesv()`.
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square or if the RHS is incompatible with the
    LHS, a `runtime_error` exception is thrown. If the inputs are distributed
    on different grids, a `runtime_exception` is thrown. If an allocation
    fails, a `bad_alloc` exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
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
    fml::scalapack::gesv(n, y.ncols(), x.data_ptr(), x.desc_ptr(), p.data_ptr(), y.data_ptr(), y.desc_ptr(), &info);
    fml::linalgutils::check_info(info, "gesv");
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void qr_internals(const bool pivot, mpimat<REAL> &x, cpuvec<REAL> &qraux, cpuvec<REAL> &work)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      const int *descx = x.desc_ptr();
      
      int info = 0;
      qraux.resize(minmn);
      
      REAL tmp;
      if (pivot)
        fml::scalapack::geqpf(m, n, NULL, descx, NULL, NULL, &tmp, -1, &info);
      else
        fml::scalapack::geqrf(m, n, NULL, descx, NULL, &tmp, -1, &info);
      
      int lwork = std::max((int) tmp, 1);
      if (lwork > work.size())
        work.resize(lwork);
      
      if (pivot)
      {
        cpuvec<int> p(n);
        p.fill_zero();
        fml::scalapack::geqpf(m, n, x.data_ptr(), descx, p.data_ptr(),
          qraux.data_ptr(), work.data_ptr(), lwork, &info);
      }
      else
        fml::scalapack::geqrf(m, n, x.data_ptr(), descx, qraux.data_ptr(),
          work.data_ptr(), lwork, &info);
      
      if (info != 0)
      {
        if (pivot)
          fml::linalgutils::check_info(info, "geqpf");
        else
          fml::linalgutils::check_info(info, "geqrf");
      }
    }
  }
  
  /**
    @brief Computes the QR decomposition.
    
    @details The factorization works mostly in-place by modifying the input
    data. After execution, the matrix will be the LAPACK-like compact QR
    representation.
    
    @param[in] pivot Should the factorization use column pivoting?
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] qraux Auxiliary data for compact QR.
    
    @impl Uses the ScaLAPACK function `pXgeqpf()` if pivoting and `pXgeqrf()`
    otherwise.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr(const bool pivot, mpimat<REAL> &x, cpuvec<REAL> &qraux)
  {
    cpuvec<REAL> work;
    qr_internals(pivot, x, qraux, work);
  }
  
  /**
    @brief Recover the Q matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[in] qraux Auxiliary data for compact QR.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl Uses the ScaLAPACK function `pXormqr()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_Q(const mpimat<REAL> &QR, const cpuvec<REAL> &qraux, mpimat<REAL> &Q, cpuvec<REAL> &work)
  {
    check_grid(QR, Q);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    const int *descQR = QR.desc_ptr();
    
    Q.resize(m, n);
    Q.fill_eye();
    const int *descQ = Q.desc_ptr();
    
    int info = 0;
    REAL tmp;
    fml::scalapack::ormqr('L', 'N', m, minmn, n, NULL, descQR,
      NULL, NULL, descQ, &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::ormqr('L', 'N', m, minmn, n, QR.data_ptr(), descQR,
      qraux.data_ptr(), Q.data_ptr(), descQ, work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "ormqr");
  }
  
  /**
    @brief Recover the R matrix from a QR decomposition.
    
    @param[in] QR The compact QR factorization, as computed via `qr()`.
    @param[out] R The R matrix.
    
    @impl Uses the ScaLAPACK function `pXlacpy()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void qr_R(const mpimat<REAL> &QR, mpimat<REAL> &R)
  {
    check_grid(QR, R);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    
    R.resize(n, n);
    R.fill_zero();
    fml::scalapack::lacpy('U', m, n, QR.data_ptr(), QR.desc_ptr(), R.data_ptr(),
      R.desc_ptr());
  }
  
  
  
  /**
    @brief Computes the singular value decomposition for tall/skinny data.
    The number of rows must be greater than the number of columns. If the number
    of rows is not significantly larger than the number of columns, this may not
    be more efficient than simply calling `linalg::svd()`.
    
    @details The operation works by computing a QR and then taking the SVD of
    the R matrix. The left singular vectors are Q times the left singular
    vectors from R's SVD, and the singular value and the right singular vectors
    are those from R's SVD.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `linalg::qr()` and `linalg::svd()`, and if computing the
    left/right singular vectors, `linalg::qr_R()` and `linalg::qr_Q()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void tssvd(mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    check_grid(x, u);
    check_grid(x, vt);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    grid g = x.get_grid();
    
    cpuvec<REAL> qraux(n);
    cpuvec<REAL> work(m);
    qr_internals(false, x, qraux, work);
    
    mpimat<REAL> R(g, n, n, x.bf_rows(), x.bf_cols());
    qr_R(x, R);
    
    mpimat<REAL> u_R(g, n, n, x.bf_rows(), x.bf_cols());
    svd(R, s, u_R, vt);
    
    u.resize(m, n);
    u.fill_eye();
    
    qr_Q(x, qraux, u, work);
    
    matmult(false, false, (REAL)1.0, u, u_R, x);
    mpihelpers::mpi2mpi(x, u);
  }
  
  /// \overload
  template <typename REAL>
  void tssvd(mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    grid g = x.get_grid();
    s.resize(n);
    
    cpuvec<REAL> qraux(n);
    cpuvec<REAL> work(m);
    qr_internals(false, x, qraux, work);
    
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const len_t mb = x.bf_rows();
    const len_t nb = x.bf_cols();
    REAL *x_d = x.data_ptr();
    
    for (len_t j=0; j<n_local; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<n_local; i++)
      {
        const int gi = fml::bcutils::l2g(i, mb, g.nprow(), g.myrow());
        const int gj = fml::bcutils::l2g(j, nb, g.npcol(), g.mycol());
        
        if (gi > gj)
          x_d[i + m_local*j] = (REAL) 0.0;
      }
    }
    
    int info = 0;
    
    REAL tmp;
    fml::scalapack::gesvd('N', 'N', n, n, x_d, x.desc_ptr(), s.data_ptr(), NULL,
      NULL, NULL, NULL, &tmp, -1, &info);
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::gesvd('N', 'N', n, n, x_d, x.desc_ptr(), s.data_ptr(), NULL,
      NULL, NULL, NULL, work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "gesvd");
  }
}


#endif
