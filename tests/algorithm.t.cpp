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

#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include "maikel/hmm/hidden_markov_model.h"
#include "maikel/hmm/algorithm.h"

namespace {

CASE ( "Do we see if a bijective map is bijective onto values?" ) {
  using Index = uint8_t;
  std::map<int, Index> bijective_map_int { {0,0}, {1,1} };
  std::map<std::string, Index> bijective_map_string { {"foo",1}, {"bar",0} };
  std::map<int, Index> not_bijective_map_int1 { {0,1}, {1,2} };
  std::map<int, Index> not_bijective_map_int2 { {0,1}, {2,1} };

  EXPECT(maikel::is_bijective_index_map(bijective_map_int));
  EXPECT(maikel::is_bijective_index_map(bijective_map_string));
  EXPECT_NOT(maikel::is_bijective_index_map(not_bijective_map_int1));
  EXPECT_NOT(maikel::is_bijective_index_map(not_bijective_map_int2));
}

CASE ("convert symbols to indicies") {
  using Index = int;
  std::vector<std::string> symbols { "foo", "bar" };
  auto symbols_to_index = maikel::map_from_symbols<Index>(ranges::view::all(symbols));
  EXPECT(symbols_to_index["foo"] == 0);
  EXPECT(symbols_to_index["bar"] == 1);

  std::vector<int> symbols_2 { 1, 2 };
  auto sti = maikel::map_from_symbols<Index>(symbols_2);
  EXPECT(sti[1] == 0);
  EXPECT(sti[2] == 1);
}

CASE ( "Test forward algorithm for test case in Rabiners Paper" ) {
  Eigen::Matrix3f A;
  A << 0.4, 0.3, 0.3,
       0.2, 0.6, 0.2,
       0.1, 0.1, 0.8;
  Eigen::Matrix3f B;
  B << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  Eigen::Vector3f pi;
  pi << 0.0, 0.0, 1.0;
  std::vector<int> sequence { 2, 2, 2, 0, 0, 2, 1, 2 };

  std::vector<float> scaling;
  std::vector<Eigen::VectorXf> alphas;
  maikel::hmm::hidden_markov_model<float> hmm(A, B, pi);
  forward(hmm, sequence, std::back_inserter(alphas), std::back_inserter(scaling));
  float probability = std::accumulate(scaling.begin(), scaling.end(), 1.0, std::multiplies<float>());
  probability = 1/probability;
  EXPECT(maikel::almost_equal<float>(probability,(1.536/10000),1));
}



CASE ( "Test forward and backward algorithms for test case in Rabiners Paper" ) {
  Eigen::Matrix3f A;
  A << 0.4, 0.3, 0.3,
       0.2, 0.6, 0.2,
       0.1, 0.1, 0.8;
  Eigen::Matrix3f B;
  B << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  Eigen::Vector3f pi;
  pi << 0.0, 0.0, 1.0;
  std::vector<int> sequence { 2, 2, 2, 0, 0, 2, 1, 2 };

  maikel::hmm::hidden_markov_model<float> hmm(A, B, pi);

  using vector_type = decltype(hmm)::vector_type;
  std::vector<float> scaling;
  std::vector<vector_type> alphas;
  maikel::hmm::forward(hmm, sequence.begin(), sequence.end(), std::back_inserter(alphas), std::back_inserter(scaling));
  float probability = std::accumulate(scaling.begin(), scaling.end(), 1.0, std::multiplies<float>());
  probability = 1/probability;
  EXPECT(maikel::almost_equal<float>(probability,(1.536/10000),1));

  std::vector<vector_type> betas_ranges;
  maikel::hmm::backward(hmm,
      sequence | ranges::view::reverse,
       scaling | ranges::view::reverse,
      std::back_inserter(betas_ranges));
  std::vector<vector_type> betas_not_ranges;
  maikel::hmm::backward(hmm,
      sequence.rbegin(), sequence.rend(),
      scaling.rbegin(), scaling.rend(),
      std::back_inserter(betas_not_ranges));
}

CASE ( "baum-welch algorithm for test case in Rabiners Paper" ) {
  Eigen::Matrix3f A;
  A << 0.4, 0.3, 0.3,
       0.2, 0.6, 0.2,
       0.1, 0.1, 0.8;
  Eigen::Matrix3f B;
  B << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  Eigen::Vector3f pi;
  pi << 0.0, 0.0, 1.0;
  std::vector<int> sequence { 2, 2, 2, 0, 0, 2, 1, 2 };

  maikel::hmm::hidden_markov_model<float> initial_hmm(A, B, pi);
  EXPECT_NO_THROW (
      auto new_hmm = maikel::hmm::naive::baum_welch(initial_hmm, sequence);
//      std::cerr << "A\n" << new_hmm.A << "\nB\n" << new_hmm.B << "\npi\n" << new_hmm.pi << std::endl;
  );
}


}


