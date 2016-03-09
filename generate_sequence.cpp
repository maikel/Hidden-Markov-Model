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

#include <fstream>
#include <iostream>

#include <boost/iterator/function_input_iterator.hpp>

#include "hmm/hidden_markov_model.h"
#include "hmm/sequence_generator.h"
#include "hmm/io.h"

enum Exit_Error_Codes {
  exit_success = 0,
  exit_not_enough_arguments = 1,
  exit_io_error = 2,
  exit_argument_error = 3
};

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence-length>\n";
    return exit_not_enough_arguments;
  }
  std::ifstream input(argv[1]);
  input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  auto model = maikel::hmm::read_hidden_markov_model<float>(input);
  input.close();

  // Try to read the HMM-model that will be used to generate a random sequence.

  std::istringstream obslen_converter(argv[2]);
  std::size_t obslen;
  if (!(obslen_converter >> obslen) || obslen == 0) {
    std::cerr << "Could not convert sequence length to std::size_t.\n";
    return exit_argument_error;
  }

  using Index = decltype(model)::size_type;
  std::function<Index()> generator = maikel::hmm::make_sequence_generator(model);
  std::cout << obslen << "\n";
  std::copy(boost::make_function_input_iterator(generator, std::size_t{0}),
            boost::make_function_input_iterator(generator, obslen),
            std::ostream_iterator<Index>(std::cout, " "));
  std::cout << std::endl;

  return exit_success;
}
