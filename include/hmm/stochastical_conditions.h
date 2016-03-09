/*
 * Copyright 2016 Maikel Nadolski
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HMM_MATRIX_TRAITS_H_
#define HMM_MATRIX_TRAITS_H_

#include "hmm/matrix.h"

#include <cmath>       // std::abs
#include <type_traits> // std::enable_if
#include <limits>      // std::numeric_limits
#include <Eigen/Dense> // Eigen::Array

namespace maikel {

  template <class T, std::size_t ulp>
      // requires FloatingPoint<T>
    inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    almost_equal(T x, T y) noexcept
    {
      return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp;
    }

  namespace hmm {

    template <class Derived, std::size_t ulp = 10000>
      inline bool is_probability_array(const Eigen::ArrayBase<Derived>& array) noexcept
      {
        using Scalar = typename Eigen::ArrayBase<Derived>::Scalar;
        bool is_not_negative = (array >= 0).all();
        bool is_normed_to_one = almost_equal<Scalar, ulp>(array.sum(), 1.0);
        return is_not_negative && is_normed_to_one;
      }

    template <class Derived, std::size_t ulp = 10000>
      bool rows_are_probability_arrays(const Eigen::DenseBase<Derived>& dense)
      {
        using Index = typename Eigen::DenseBase<Derived>::Index;
        Index rows = dense.rows();
        for (Index i = 0; i < rows; ++i)
          if (!is_probability_array(dense.row(i).array()))
            return false;
        return true;
      }
  }
}



#endif /* TYPE_TRAITS_H_ */
