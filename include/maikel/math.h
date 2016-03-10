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

#ifndef MATH_H_
#define MATH_H_

#include <type_traits>
#include <cmath>
#include <limits>

namespace maikel {

  template <class T, std::size_t ulp = 1>
    // requires FloatingPoint<T>
    inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    almost_equal(T x, T y) noexcept
    {
      return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp;
    }

}


#endif /* MATH_H_ */
