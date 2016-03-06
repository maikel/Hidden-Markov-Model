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

#include "hidden-markov-models.t.h"

#include "type_traits.h"

#include <fstream>
#include <Eigen/Dense>

namespace {

CASE( "Read array data in 'array.dat' and its same as hard coded." )
{
  std::ifstream datafile("array.dat");
  EXPECT(datafile);
  Eigen::ArrayXf array_hardcoded(3);
  array_hardcoded << 0.3, 0.3, 0.4;
  Eigen::ArrayXf array_from_file(3);
  std::copy(std::istream_iterator<float>(datafile),
            std::istream_iterator<float>(),
            array_from_file.data());
  EXPECT(array_hardcoded.size() == array_from_file.size());
  for (int i = 0; i < array_from_file.size(); ++i)
    EXPECT(array_from_file(i) == array_hardcoded(i));
}

CASE( "is stochastic or not" )
{
  Eigen::ArrayXf array(3);
  array << 0.5, 0.2, 0.3;
  EXPECT(maikel::is_probability_array(array));
}

CASE( "is stochastic matrix ")
{
  Eigen::MatrixXf matrix(2,2);
  matrix << 0.3, 0.7,
            0.5, 0.5;
  EXPECT(maikel::rows_are_probability_arrays(matrix));
}

}
