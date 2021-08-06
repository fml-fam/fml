// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML__INTERNALS_DIMOPS_H
#define FML__INTERNALS_DIMOPS_H
#pragma once


namespace fml
{
namespace dimops
{
  enum sweep_op
  {
    /**
      Arithmetic operations for `dimops::rowsweep()` and `dimops::colsweep()`
      functions. These operate by either row or column-wise taking a matrix and
      applying an appropriately sized vector to the entries of that matrix by
      addition, subtraction, multiplication, or division.
    */
    SWEEP_ADD, SWEEP_SUB, SWEEP_MUL, SWEEP_DIV
  };
}
}


#endif
