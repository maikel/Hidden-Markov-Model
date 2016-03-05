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
#include "../types.h"

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
        void forward(
            ObInputIter ob_start, ObInputIter ob_end,                   // inputs
            AlphaOutputIter alphaout, ScalingOutputIter scalout)        // outputs
        const
        {
          if (ob_start == ob_end)
            return;

          // get first observation and cacluate alpha_0
          index_type ob = gsl::narrow<index_type>(*ob_start);
          assert(0 <= ob && ob < symbols());
          vector_type alpha_0(states());
          float_type scaling { 0 };
          for (index_type i = 0; i < states(); ++i) {
            alpha_0(i) = pi(i)*B(i,ob);
            scaling += alpha_0(i);
          }
          assert(scaling > 0);
          assert(almost_equal<float_type>(scaling, alpha_0.sum(), 1));
          scaling = 1/scaling;
          alpha_0 *= scaling;
          assert(almost_equal<float_type>(alpha_0.sum(), 1.0, 1));

          // write data to output iterators and go to next observation
          *alphaout = alpha_0;
          *scalout  = scaling;
          ++alphaout;
          ++scalout;
          ++ob_start;

          // start recursion
          forward_with_initial(ob_start, ob_end, alphaout, scalout, alpha_0);
        }

      template <class ObInputIter, class AlphaOutputIter, class ScalingOutputIter>
        void forward_with_initial(
            ObInputIter ob_start, ObInputIter ob_end,              // inputs
            AlphaOutputIter alphaout, ScalingOutputIter scalout,   // outputs
            vector_type const& prev_alpha)                        // algorithm start data
        const
        {
          if (ob_start == ob_end)
            return;

          vector_type alpha(states());
          while (!(ob_start == ob_end)) {
            index_type ob = gsl::narrow<index_type>(*ob_start);
            assert(0 <= ob && ob < symbols());

            // do the recursion
            float_type scaling { 0 };
            for (index_type j = 0; j < states(); ++j) {
              alpha(j) = 0.0;
              for (index_type i = 0; i < states(); ++i)
                alpha(j) += prev_alpha(i)*A(i,j);
              alpha(j) *= B(j,ob);
              scaling += alpha(j);
            }

            // scaling with assertions
            assert(scaling > 0);
            assert(almost_equal<float_type>(scaling, alpha.sum(), 1));
            scaling = !scaling ? 0 : 1 / scaling;
            alpha *= scaling;
            assert(almost_equal<float_type>(alpha.sum(), 1.0, 1));

            // write to output
            *alphaout = alpha;
            alpha.swap(prev_alpha);
            *scalout = scaling;
            ++alphaout;
            ++scalout;
            ++ob_start;
          }
        }

      template <class ObInputIter, class ScalingInputIter, class BetaOutputIter>
        void backward(ObInputIter ob_start, ObInputIter ob_end, ScalingInputIter scalit, BetaOutputIter betaout)
        const
        {
          if (ob_start == ob_end)
            return;

          // calculate initial beta_T
          float_type scaling = *scalit;
          assert(scaling > 0);
          vector_type beta(states());
          for (index_type i = 0; i < states(); ++i)
            beta(i) = scaling;

          // write beta_T to output iterator and advance iterators
          *betaout = beta;
          ++betaout;
          ++scalit;

          // go for recursion
          backward_with_initial(ob_start, ob_end, scalit, betaout, beta);
        }

      template <class ObInputIter, class ScalingInputIter, class BetaOutputIter>
        void backward_with_initial(
            ObInputIter ob_start, ObInputIter ob_end, ScalingInputIter scalit,
            BetaOutputIter betaout,
            vector_type const& next_beta)
        const
        {
          if (ob_start == ob_end)
            return;

          vector_type beta(states());
          while (ob_start+1 != ob_end) {
            index_type ob = gsl::narrow<index_type>(*ob_start);
            assert(0 <= ob && ob < symbols());

            // do the recursion
            float scaling = *scalit;
            assert(scaling > 0);
            for (index_type i = 0; i < states(); ++i) {
              beta(i) = 0.0;
              for (index_type j = 0; j < states(); ++j)
                beta(i) += A(i,j)*B(j,ob)*next_beta(j);
              beta(i) *= scaling;
            }

            // write to output and increment iterators
            *betaout = beta;
            beta.swap(next_beta);
            ++betaout;
            ++ob_start;
            ++scalit;
          }
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
