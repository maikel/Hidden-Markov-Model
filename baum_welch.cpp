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

#include <iostream>
#include <fstream>
#include <iterator>

#include <maikel/hmm/algorithm.h>
#include <maikel/hmm/io.h>

using namespace std;
using namespace maikel;
using row_vector = hmm::hidden_markov_model<double>::row_vector;

double calculate_alpha(vector<uint8_t> const& sequence, hmm::hidden_markov_model<double> const& hmm,
    vector<double>& scaling, vector<row_vector>& alphas)
{
  double logprob = 0;
  size_t count = 0;
  for (pair<double, row_vector> const& sal
      : hmm::forward(begin(sequence), end(sequence), hmm)) {
    scaling[count] = sal.first;
    alphas[count]  = sal.second;
    logprob += log(sal.first);
    ++count;
  }
  return logprob;
}

void calculate_beta(vector<uint8_t> const& sequence, hmm::hidden_markov_model<double> const& hmm,
    vector<double> const& scaling, vector<row_vector>& betas)
{
  size_t count = 0;
  for (row_vector const& beta
      : hmm::backward(sequence.rbegin(), sequence.rend(), scaling.rbegin(), hmm)) {
    betas[sequence.size()-count-1] = beta;
    ++count;
  }
}

template <class Update>
void update_hmm(
    Update& update,
    vector<uint8_t>& sequence,
    vector<row_vector>& alphas,
    vector<row_vector>& betas,
    vector<double>& scaling,
    row_vector& pi,
    hmm::hidden_markov_model<double>& hmm)
{
  auto matrices = update(begin(sequence), end(sequence), begin(alphas), begin(betas), scaling.back(), hmm);
  pi = alphas[0].cwiseProduct(betas[0]) / scaling[0];
  //    cout << "step #" << step << " log P(O|lambda): " << -logprob << "\n";
  //    cout << matrices.first.format(Eigen::IOFormat(5)) << endl;
  hmm = hmm::hidden_markov_model<double>(matrices.first, matrices.second, pi);
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
    return 1;
  }

  // read data
  ifstream model_input(argv[1]);
  auto hmm = hmm::read_hidden_markov_model<double>(model_input);
  ifstream sequence_input(argv[2]);
  vector<uint8_t> sequence = hmm::read_sequence<uint8_t>(sequence_input);
  vector<double> scaling(sequence.size());
  vector<row_vector> alphas(sequence.size());
  vector<row_vector> betas(sequence.size());
  row_vector pi(hmm.states());

  size_t step = 0;
  double logprob_old = 0, logprob = 0;
  auto update = hmm::update_matrices<
      decltype(sequence)::iterator,
        decltype(alphas)::iterator,
         decltype(betas)::iterator,
                            double>(hmm.states(), hmm.symbols());

  cout.flags(ios_base::fixed);
  do {
    ++step;
    // calculate alphas
    swap(logprob_old, logprob);
    logprob = calculate_alpha(sequence, hmm, scaling, alphas);

    // calculate betas
    calculate_beta(sequence, hmm, scaling, betas);

    update_hmm(update, sequence, alphas, betas, scaling, pi, hmm);
  } while (!almost_equal<double,100>(logprob, logprob_old));

  cout << "steps: " << step << ", A:\n" << hmm.transition_matrix() << endl;
}
