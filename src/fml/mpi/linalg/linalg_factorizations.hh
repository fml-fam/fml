// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_FACTORIZATIONS_H
#define FML_MPI_LINALG_LINALG_FACTORIZATIONS_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../copy.hh"
#include "../mpimat.hh"

#include "linalg_blas.hh"
#include "linalg_err.hh"
#include "scalapack.hh"


namespace fml
{
namespace linalg
{
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
      else // if (nu <= minmn && nv <= minmn)
      {
        jobu = 'V';
        jobvt = 'V';
        
        const int mb = x.bf_rows();
        const int nb = x.bf_cols();
        
        u.resize(m, minmn, mb, nb);
        vt.resize(minmn, n, mb, nb);
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
    err::check_grid(x, u);
    err::check_grid(x, vt);
    
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
    err::check_grid(x, vectors);
    
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
    int lwork = std::max(1, (int)tmp);
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
    err::check_grid(x, y);
    
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
    err::check_grid(QR, Q);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    const int *descQR = QR.desc_ptr();
    
    Q.resize(m, minmn);
    Q.fill_eye();
    const int *descQ = Q.desc_ptr();
    
    int info = 0;
    REAL tmp;
    fml::scalapack::ormqr('L', 'N', m, minmn, minmn, NULL, descQR,
      NULL, NULL, descQ, &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::ormqr('L', 'N', m, minmn, minmn, QR.data_ptr(), descQR,
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
    err::check_grid(QR, R);
    
    const len_t m = QR.nrows();
    const len_t n = QR.ncols();
    const len_t minmn = std::min(m, n);
    
    R.resize(minmn, n);
    R.fill_zero();
    fml::scalapack::lacpy('U', m, n, QR.data_ptr(), QR.desc_ptr(), R.data_ptr(),
      R.desc_ptr());
  }
  
  
  
  namespace
  {
    template <typename REAL>
    void lq_internals(mpimat<REAL> &x, cpuvec<REAL> &lqaux, cpuvec<REAL> &work)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      const len_t minmn = std::min(m, n);
      
      const int *descx = x.desc_ptr();
      
      int info = 0;
      lqaux.resize(minmn);
      
      REAL tmp;
      fml::scalapack::gelqf(m, n, NULL, descx, NULL, &tmp, -1, &info);
      int lwork = std::max((int) tmp, 1);
      if (lwork > work.size())
        work.resize(lwork);
      
      fml::scalapack::gelqf(m, n, x.data_ptr(), descx, lqaux.data_ptr(),
        work.data_ptr(), lwork, &info);
      
      if (info != 0)
        fml::linalgutils::check_info(info, "gelqf");
    }
  }
  
  /**
    @brief Computes the LQ decomposition.
    
    @details The factorization works mostly in-place by modifying the input
    data. After execution, the matrix will be the LAPACK-like compact LQ
    representation.
    
    @param[inout] x Input data matrix. Values are overwritten.
    @param[out] lqaux Auxiliary data for compact LQ.
    
    @impl Uses the ScaLAPACK function `pXgelqf()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq(mpimat<REAL> &x, cpuvec<REAL> &lqaux)
  {
    cpuvec<REAL> work;
    lq_internals(x, lqaux, work);
  }
  
  /**
    @brief Recover the L matrix from a LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[out] L The L matrix.
    
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
  void lq_L(const mpimat<REAL> &LQ, mpimat<REAL> &L)
  {
    err::check_grid(LQ, L);
    
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    L.resize(m, minmn);
    L.fill_zero();
    
    fml::scalapack::lacpy('L', m, n, LQ.data_ptr(), LQ.desc_ptr(), L.data_ptr(),
      L.desc_ptr());
  }
  
  /**
    @brief Recover the Q matrix from a LQ decomposition.
    
    @param[in] LQ The compact LQ factorization, as computed via `lq()`.
    @param[in] lqaux Auxiliary data for compact LQ.
    @param[out] Q The Q matrix.
    @param[out] work Workspace array. Will be resized as necessary.
    
    @impl Uses the ScaLAPACK function `pXormlq()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lq_Q(const mpimat<REAL> &LQ, const cpuvec<REAL> &lqaux, mpimat<REAL> &Q, cpuvec<REAL> &work)
  {
    err::check_grid(LQ, Q);
    
    const len_t m = LQ.nrows();
    const len_t n = LQ.ncols();
    const len_t minmn = std::min(m, n);
    
    const int *descLQ = LQ.desc_ptr();
    
    Q.resize(minmn, n);
    Q.fill_eye();
    const int *descQ = Q.desc_ptr();
    
    int info = 0;
    REAL tmp;
    fml::scalapack::ormlq('R', 'N', minmn, n, minmn, NULL, descLQ,
      NULL, NULL, descQ, &tmp, -1, &info);
    
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::ormlq('R', 'N', minmn, n, minmn, LQ.data_ptr(), descLQ,
      lqaux.data_ptr(), Q.data_ptr(), descQ, work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "ormlq");
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
    err::check_grid(x, u);
    err::check_grid(x, vt);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    const grid g = x.get_grid();
    
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
    copy::mpi2mpi(x, u);
  }
  
  /// \overload
  template <typename REAL>
  void tssvd(mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    if (m <= n)
      throw std::runtime_error("'x' must have more rows than cols");
    
    const grid g = x.get_grid();
    s.resize(n);
    
    cpuvec<REAL> qraux(n);
    cpuvec<REAL> work(m);
    qr_internals(false, x, qraux, work);
    
    fml::mpi_utils::tri2zero('L', false, g, n, n, x.data_ptr(), x.desc_ptr());
    
    int info = 0;
    
    REAL tmp;
    fml::scalapack::gesvd('N', 'N', n, n, x.data_ptr(), x.desc_ptr(),
      s.data_ptr(), NULL, NULL, NULL, NULL, &tmp, -1, &info);
    int lwork = (int) tmp;
    if (lwork > work.size())
      work.resize(lwork);
    
    fml::scalapack::gesvd('N', 'N', n, n, x.data_ptr(), x.desc_ptr(),
      s.data_ptr(), NULL, NULL, NULL, NULL, work.data_ptr(), lwork, &info);
    fml::linalgutils::check_info(info, "gesvd");
  }
  
  
  
  /**
    @brief Computes the singular value decomposition using the
    "crossproducts SVD". This method is not numerically stable.
    
    @details The operation works by computing the crossproducts matrix X^T * X
    or X * X^T (whichever is smaller) and then computing the eigenvalue
    decomposition. 
    
    @param[inout] x Input data matrix.
    @param[out] s Vector of singular values.
    @param[out] u Matrix of left singular vectors.
    @param[out] vt Matrix of (transposed) right singular vectors.
    
    @impl Uses `crossprod()` or `tcrossprod()` (whichever is smaller), and
    `eigen_sym()`.
    
    @allocs If the any outputs are inappropriately sized, they will
    automatically be re-allocated. Additionally, some temporary work storage
    is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void cpsvd(const mpimat<REAL> &x, cpuvec<REAL> &s, mpimat<REAL> &u, mpimat<REAL> &vt)
  {
    err::check_grid(x, u);
    err::check_grid(x, vt);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const len_t minmn = std::min(m, n);
    
    const grid g = x.get_grid();
    mpimat<REAL> cp(g, x.bf_rows(), x.bf_cols());
    
    if (m >= n)
    {
      crossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, vt);
      vt.rev_cols();
      copy::mpi2mpi(vt, cp);
    }
    else
    {
      tcrossprod((REAL)1.0, x, cp);
      eigen_sym(cp, s, u);
      u.rev_cols();
      copy::mpi2mpi(u, cp);
    }
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
    
    REAL *ev_d;
    if (m >= n)
      ev_d = vt.data_ptr();
    else
      ev_d = cp.data_ptr();
    
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const len_t mb = x.bf_rows();
    const len_t nb = x.bf_cols();
    for (len_t j=0; j<n_local; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<m_local; i++)
      {
        const int gi = fml::bcutils::l2g(i, mb, g.nprow(), g.myrow());
        const int gj = fml::bcutils::l2g(j, nb, g.npcol(), g.mycol());
        
        if (gi < minmn && gj < minmn)
          ev_d[i + m_local*j] /= s_d[gj];
      }
    }
    
    if (m >= n)
    {
      matmult(false, false, (REAL)1.0, x, vt, u);
      xpose(cp, vt);
    }
    else
      matmult(true, false, (REAL)1.0, cp, x, vt);
  }
  
  /// \overload
  template <typename REAL>
  void cpsvd(const mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    const grid g = x.get_grid();
    mpimat<REAL> cp(g, x.bf_rows(), x.bf_cols());
    
    if (m >= n)
      crossprod((REAL)1.0, x, cp);
    else
      tcrossprod((REAL)1.0, x, cp);
    
    eigen_sym(cp, s);
    
    s.rev();
    REAL *s_d = s.data_ptr();
    #pragma omp for simd
    for (len_t i=0; i<s.size(); i++)
      s_d[i] = sqrt(fabs(s_d[i]));
  }
  
  
  
  /**
    @brief Compute the Choleski factorization.
    
    @details The matrix should be 1. square, 2. symmetric, 3. positive-definite.
    Failure of any of these conditions can lead to a runtime exception. The
    input is replaced by its lower-triangular Choleski factor.
    
    @param[inout] x Input data matrix, replaced by its lower-triangular Choleski
    factor.
    
    @impl Uses the ScaLAPACK function `pXpotrf()`.
    
    @allocs Some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void chol(mpimat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (n != x.ncols())
      throw std::runtime_error("'x' must be a square matrix");
    
    int info = 0;
    fml::scalapack::potrf('L', n, x.data_ptr(), x.desc_ptr(), &info);
    
    if (info < 0)
      fml::linalgutils::check_info(info, "potrf");
    else if (info > 0)
      throw std::runtime_error("chol: leading minor of order " + std::to_string(info) + " is not positive definite");
    
    fml::mpi_utils::tri2zero('U', false, x.get_grid(), n, n, x.data_ptr(), x.desc_ptr());
  }
}
}


#endif
