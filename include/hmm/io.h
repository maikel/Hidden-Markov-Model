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


#ifndef HMM_IO_H_
#define HMM_IO_H_

#include <map>
#include <istream>
#include <Eigen/Dense>
#include <gsl_assert.h>
#include <range/v3/all.hpp>

#include "hidden_markov_model.h"

namespace maikel { namespace hmm {

  struct getline_error: public std::runtime_error {
      getline_error(std::string s): std::runtime_error(s) {}
  };

  struct read_ascii_matrix_error: public std::runtime_error {
      read_ascii_matrix_error(std::string s): std::runtime_error(s) {}
  };

  struct read_sequence_error: public std::runtime_error {
      read_sequence_error(std::string s): std::runtime_error(s) {}
  };

  inline std::istream& getline(std::istream& in, std::istringstream& linestream)
  {
    Expects(in);
    std::string line;
    if (!std::getline(in, line))
      throw getline_error("Could not read the line from given stream.");
    linestream.str(line);
    linestream.clear();
    return in;
  }

  template <class size_type>
    std::pair<size_type, size_type> getdims(std::istream& in)
    {
      Expects(in);
      std::pair<size_type, size_type> dim;
      std::istringstream line;
      getline(in, line);
      if (!(line >> dim.first >> dim.second))
        throw read_ascii_matrix_error("Could not read dimensions.");
      return dim;
    }

  template <class float_type>
    typename std::enable_if<
        std::is_floating_point<float_type>::value,
    Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic>>::type
    read_ascii_matrix(std::istream& in, size_t rows, size_t cols)
    {
      Expects(in);
      MatrixX<float_type> matrix(rows, cols);
      std::istringstream line;
      for (size_t i = 0; i < rows; ++i) {
        getline(in, line);
        for (size_t j = 0; j < cols; ++j)
          if (!(line >> matrix(i,j)))
            throw read_ascii_matrix_error("Could not read entries in line: " + line.str() + ".");
      }
      Ensures(gsl::narrow<size_t>(matrix.rows()) == rows &&
              gsl::narrow<size_t>(matrix.cols()) == cols);
      return matrix;
    }

  template <class float_type>
    typename std::enable_if<
        std::is_floating_point<float_type>::value,
    hidden_markov_model<float_type>>::type
    read_hidden_markov_model(std::istream& in)
    {
      Expects(in);
      using matrix = typename hidden_markov_model<float_type>::matrix_type;
      using index = typename hidden_markov_model<float_type>::index_type;
      index states;
      index symbols;
      std::tie(states, symbols) = getdims<index>(in);
      matrix A  = read_ascii_matrix<float_type>(in, states, states);
      matrix B  = read_ascii_matrix<float_type>(in, states, symbols);
      matrix pi = read_ascii_matrix<float_type>(in, 1, states);
      Ensures(A.rows() == states && A.cols() == states);
      Ensures(B.rows() == states && B.cols() == symbols);
      Ensures(pi.rows() == 1 && pi.cols() == states);
      return hidden_markov_model<float_type>(A, B, pi);
    }

  template <class float_type>
    typename std::enable_if<
        std::is_floating_point<float_type>::value,
    void>::type
    print_model_parameters(std::ostream& out, hidden_markov_model<float_type> const& model)
    {
      out << "epsilon: " << std::numeric_limits<float_type>::epsilon() << "\n";
      out << "N= " << model.states() << "\n";
      out << "M= " << model.symbols() << "\n";
      out << "A:\n" << model.A << "\n";
      out << "B:\n" << model.B << "\n";
      out << "pi:\n" << model.pi << "\n";
      out << std::flush;
    }

  template <class size_type>
    size_type read_sequence_length(std::istream& in)
    {
      std::string line;
      std::getline(in, line);
      std::istringstream linestream(line);
      size_type length;
      linestream >> length;
      return length;
    }

  template <class Integral, class Symbol>
    std::vector<Integral>
    read_sequence(std::istream& in, std::map<Symbol,Integral>& symbol_to_index)
    {
      std::vector<Integral> sequence;
      sequence.reserve(read_sequence_length<std::size_t>(in));
      auto symbol_map = [&symbol_to_index] (Symbol const& s) {
          auto found = symbol_to_index.find(s);
          if (found == symbol_to_index.end())
            throw read_sequence_error("Unkown Symbols in Input.");
          return found->second;
      };
      auto sequence_input = ranges::istream_range<Symbol>(in);
      ranges::copy(sequence_input | ranges::view::transform(symbol_map), ranges::back_inserter(sequence));
      return sequence;
    }
} // namespace hmm
} // namespace maikel

#endif /* HMM_IO_H_ */
