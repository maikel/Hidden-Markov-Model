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

#include <fstream>
#include <Eigen/Dense>
#include "../include/hmm/io.h"
#include "hmm/hidden_markov_model.h"

CASE ( "Can read the text file 'A.dat' and create a dynamic allocated matrix." )
{
  std::ifstream in("A.dat");
  size_t n, m;
  std::tie(n, m) = maikel::hmm::getdims<size_t>(in);
  Eigen::MatrixXf matrix = maikel::hmm::read_ascii_matrix<float>(in, n, m);
  Eigen::MatrixXf checking(3,3);
  checking << 0.33333, 0.33333, 0.33333,
              0.4, 0.4, 0.2,
              0.5, 0.0, 0.5;
  EXPECT(checking == matrix);
}

CASE ( "Can see that model data is not stochastical." )
{
  std::ifstream in("no_good_model.dat");
  EXPECT_THROWS_AS(
      auto hmm = maikel::hmm::read_hidden_markov_model<float>(in),
      maikel::hmm::hidden_markov_model<float>::arguments_not_probability_arrays
  );
}

CASE ( "Reads a valid model" )
{
  std::ifstream in("model.dat");
  EXPECT_NO_THROW(
      auto hmm = maikel::hmm::read_hidden_markov_model<float>(in)
  );
}

