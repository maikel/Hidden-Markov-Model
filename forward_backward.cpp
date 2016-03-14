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

#include <maikel/hmm/hidden_markov_model.h>
#include <maikel/hmm/algorithm.h>
#include <maikel/hmm/io.h>

#include <gsl_util.h>

#include <iostream>
#include <fstream>

using namespace maikel;
using namespace std;
using namespace gsl;
using model = maikel::hmm::hidden_markov_model<double>;

pair<model, vector<uint8_t>>
read_model_and_sequence(
    string const& model_path,
    string const& seq_path,
    vector<int> const& symbols)
{
  MAIKEL_PROFILER;
  ifstream model_in(model_path);
  auto hmm = hmm::read_hidden_markov_model<double>(model_in);
  map<int,uint8_t> symbol_to_index = map_from_symbols<uint8_t>(symbols);
  ifstream seq_in(seq_path);
  cout << "Read sequence ...\n";
  vector<uint8_t> sequence = hmm::read_sequence(seq_in, symbol_to_index);
  return {hmm, sequence};
}

void
calculate_forward_coeff(
    vector<uint8_t> const& seq, model const& hmm, ofstream& alphas, ofstream& scaling)
{
  MAIKEL_PROFILER;
  cout << "Calculate and Write data for forward coefficients ...\n";
  for (auto&& coeff : hmm::forward(seq, hmm)) {
    double factor = coeff.first;
    model::row_vector const& alpha = coeff.second;
    scaling.write(reinterpret_cast<char*>(&factor), sizeof(factor));
    alphas.write(reinterpret_cast<const char*>(alpha.data()), sizeof(double)*hmm.states());
  }
}

void
calculate_backward_coeff(
    vector<uint8_t> const& seq, model const& hmm, vector<double>& scaling, ofstream& betas)
{
  MAIKEL_PROFILER;
  cout << "Calculate and Write data for backward coefficients ...\n";
  for (auto&& coeff : hmm::backward(begin(seq), begin(seq)+scaling.size(), begin(scaling), hmm))
    betas.write(reinterpret_cast<const char*>(coeff.data()), sizeof(double)*hmm.states());
}

vector<double>
get_reversed_chunk(ifstream& scaling, streamoff offset, streamsize max)
{
  MAIKEL_PROFILER;
  scaling.clear();
  scaling.seekg(0, ios::end);
  streamsize length = narrow<streamsize>(scaling.tellg());
  streamsize off = min(narrow<streamsize>(offset+max), length);
  off /= sizeof(double);
  off *= sizeof(double);
  scaling.clear();
  scaling.seekg(-off, ios::end);
  cout << "Reading scaling factors from negative offset: " << -off << endl;
  vector<double> factors(narrow<vector<double>::size_type>(off)/sizeof(double));
  copy(istreambuf_iterator<char>(scaling), istreambuf_iterator<char>(), reinterpret_cast<char*>(factors.data()));
  return factors;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Not enough arguments. Usage: " << argv[0] << " <model.hmm> <sequence.dat>\n";
    std::terminate();
  }

  const streamsize gigabyte = 1024*1024*1024;

  auto parameter = read_model_and_sequence(argv[1], argv[2], {0, 1});
  model& hmm = parameter.first;
  vector<uint8_t>& sequence = parameter.second;

  {
    ofstream alphas("alphas.dat", ofstream::binary);
    ofstream scaling("scaling.dat", ofstream::binary);
    calculate_forward_coeff(sequence, hmm, alphas, scaling);
  }
  {
    ifstream scaling("scaling.dat", ifstream::binary);
    ofstream betas("betas.dat", ofstream::binary);
    vector<double> chunk = get_reversed_chunk(scaling, 0, gigabyte);
    calculate_backward_coeff(sequence, hmm, chunk, betas);
  }

  function_profiler::print_statistics(cout);
}
