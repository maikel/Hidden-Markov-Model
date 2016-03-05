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

#ifndef ARITHMETIC_H_
#define ARITHMETIC_H_

#include <cmath>       // std::abs
#include <type_traits> // std::enable_if
#include <limits>      // std::numeric_limits
#include <Eigen/Dense> // Eigen::Array

namespace maikel {

  template <class float_type>
    using ArrayXX = Eigen::Array<float_type, Eigen::Dynamic, Eigen::Dynamic>;

  template <class float_type>
    using ArrayX = Eigen::Array<float_type, 1, Eigen::Dynamic>;

  template <class float_type>
    using MatrixX = Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic>;

  template <class float_type>
    using VectorX = Eigen::Matrix<float_type, 1, Eigen::Dynamic>;

  template <class T>
      // requires FloatingPoint<T>
  inline
  typename std::enable_if<
      std::is_floating_point<T>::value,
  bool>::type
      almost_equal(T x, T y, size_t ulp)
  noexcept {
    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp;
  }

}



#endif /* ARITHMETIC_H_ */
