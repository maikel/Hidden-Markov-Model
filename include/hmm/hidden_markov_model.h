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

#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <Eigen/Dense>

#include "arithmetic.h"
#include "stochastic_properties.h"
#include "hmm/algorithm.h"

namespace maikel { namespace hmm {


  template <class FloatT>
    struct hidden_markov_model
    {
      using float_type             = FloatT;
      using vector_type            = VectorX<float_type>;
      using matrix_type            = MatrixX<float_type>;
      using index_type             = typename matrix_type::Index;
      using transition_matrix_type = matrix_type; // can be different to symbols for fix-size
      using symbols_matrix_type    = matrix_type; // can be different to transition matrix for fix-size

      transition_matrix_type A;
      symbols_matrix_type B;
      vector_type pi;
      const index_type m_NumStates = 2;
      const index_type m_NumSymbols = 3;

      hidden_markov_model(
          const matrix_type& transition_matrix,
          const matrix_type& symbol_matrix,
          const vector_type& initial_dist )
       : A(transition_matrix.rows(), transition_matrix.cols()),
         B(symbol_matrix.rows(), symbol_matrix.cols()),
         pi(initial_dist.size()), m_NumStates(B.rows()), m_NumSymbols(B.cols())
      {
        A = transition_matrix;
        B = symbol_matrix;
        pi = initial_dist;
        if (!rows_are_probability_arrays(A) || !rows_are_probability_arrays(B) || !is_probability_array(pi.array()))
          throw arguments_not_probability_arrays(
              "Some inputs in constructor do not have the stochastical property.");
        if (A.rows() != A.cols() || A.rows() != B.rows() || A.rows() != pi.cols())
          throw dimensions_not_consistent(
              "Dimensions of input matrices are not consistent with each other.");
      }

      inline index_type states() const noexcept  { return m_NumStates;  }
      inline index_type symbols() const noexcept { return m_NumSymbols; }

      inline const transition_matrix_type& transition_matrix()    const noexcept { return A; }
      inline const symbols_matrix_type&    symbol_probabilities() const noexcept { return B; }
      inline const vector_type&            initial_distribution() const noexcept { return pi; }

      template <class ObInputIter, class AlphaOutputIter, class ScalingOutputIter>
        inline void forward(
            ObInputIter start, ObInputIter end,                         // inputs
            AlphaOutputIter alphaout, ScalingOutputIter scalout) const // outputs
      {
//        ::maikel::hmm::forward(start, end, alphaout, scalout, *this);
      }

      template <class ObInputIter, class ScalInputIter, class BetaOutputIter>
        inline void backward(
            ObInputIter start, ObInputIter end, ScalInputIter scalin, // inputs
            BetaOutputIter betaout) const                            // outputs
      {
//        ::maikel::hmm::backward(start, end, scalin, betaout, *this);
      }

      struct hmm_errors: public std::runtime_error {
          hmm_errors(std::string s): std::runtime_error(s) {}
      };
      struct arguments_not_probability_arrays: public hmm_errors {
          arguments_not_probability_arrays(const std::string& a):hmm_errors(a) {}
      };
      struct dimensions_not_consistent: public hmm_errors {
          dimensions_not_consistent(const std::string& a): hmm_errors(a) {}
      };

    };

} // namespace hmm
} // namespace maikel

#endif /* HIDDEN_MARKOV_MODEL_H_ */
