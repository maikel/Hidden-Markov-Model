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
 *
 *
 * This header containts different implementations of the Baum-Welch algorithm.
 * The purpose of the Baum-Welch algorithm is to determine a HMM (A,B,pi) which
 * locally maximizes the probability for generating an observed sequence
 * O_1, O_2, ..., O_N for symbols.
 */

#ifndef HMM_ALGORITHM_BAUM_WELCH_H_
#define HMM_ALGORITHM_BAUM_WELCH_H_

#include <range/v3/all.hpp>
#include <vector>
#include "hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  namespace detail { namespace baum_welch {

    template <class SeqRng, class AlphaRng, class BetaRng, class HiddenMarkovModel,
               class Matrix = typename HiddenMarkovModel::matrix_type,
               class Float = typename HiddenMarkovModel::float_type>
      std::pair<Matrix, Matrix>
      update_matrices(
          SeqRng&   sequence,
          AlphaRng& alphas,
          BetaRng&  betas,
          Float scaling,
          HiddenMarkovModel const& model)
      {
        using alpha_t = typename AlphaRng::value_type;
        using beta_t  = typename BetaRng::value_type;
        using Index   = typename HiddenMarkovModel::index_type;
        using Vector  = typename HiddenMarkovModel::vector_type;
        using namespace ranges;
        using namespace gsl;

        Expects(!empty(sequence) && !empty(betas) && !empty(alphas));

        Index states  = model.states();
        Index symbols = model.symbols();
        Matrix xi     = Matrix::Zero(states, states);
        Matrix B      = Matrix::Zero(states, symbols);
        Vector g_sum  = Vector::Zero(states);
        Vector gamma(states);

        auto seq_iterator   = begin(sequence);
        auto alpha_iterator = begin(alphas);
        auto beta_iterator  = begin(betas);
        while (seq_iterator+1 != end(sequence) &&
            alpha_iterator+1 != end(alphas) && beta_iterator+1 != end(betas)) {
          Index ob      = narrow<Index>(*seq_iterator++);
          Index ob_next = narrow<Index>(*seq_iterator);
          Expects(0 <= ob      && ob      < symbols);
          Expects(0 <= ob_next && ob_next < symbols);

          gamma.setZero();
          alpha_t alpha = *alpha_iterator++;
          beta_t beta   = *((beta_iterator++)+1);
          Expects(alpha.size() == states);
          Expects(beta.size()  == states);
          for (Index i = 0; i < states; ++i)
            for (Index j = 0; j < states; ++j) {
              float xi_t = alpha(i)*model.A(i,j)*model.B(j,ob_next)*beta(j);
              xi(i,j)  += xi_t;
              gamma(i) += xi_t;
            }
          for (Index j = 0; j < states; ++j) {
            B(j,ob) += gamma(j);
            g_sum(j) += gamma(j);
          }
        }
        for (Index j = 0; j < states; ++j)
          for (Index i = 0; i < states; ++i)
            xi(i,j) /= g_sum(i);
        Ensures(maikel::rows_are_probability_arrays(xi));

        Expects(seq_iterator   != end(sequence));
        Expects(alpha_iterator != end(alphas));
        Expects(beta_iterator  != end(betas));
        Index ob = narrow<Index>(*seq_iterator++);
        alpha_t alpha = *alpha_iterator++;
        beta_t  beta  = *beta_iterator++;
        for (Index i = 0; i < states; ++i) {
          float entry = alpha(i)*beta(i) / scaling;
          B(i,ob) += entry;
          g_sum(i) += entry;
        }
        for (Index i = 0; i < states; ++i)
          for (Index k = 0; k < symbols; ++k)
            B(i,k) /= g_sum(i);
        Ensures(maikel::rows_are_probability_arrays(B));
        Ensures(seq_iterator == end(sequence));
        Ensures(alpha_iterator == end(alphas));
        Ensures(beta_iterator == end(betas));

        return std::make_pair(xi, B);
      }
  } }

  namespace naive {

    /**
     * naive implementation of the baum-welch algorithm. just save everything
     * into std::vector's and do not buffer anything in files. Just see if it
     * works.
     */
    template <class float_type, class SeqRng>
      hidden_markov_model<float_type>
      baum_welch(
          hidden_markov_model<float_type> const& initial_model,
          SeqRng sequence)
      {
        // typedefs
        using vector_type = typename hidden_markov_model<float_type>::vector_type;
        using matrix_type = typename hidden_markov_model<float_type>::matrix_type;
        using index_type  = typename hidden_markov_model<float_type>::index_type;

        size_t T = sequence.size();

        // allocate memory
        std::vector<float> scaling;
        std::vector<vector_type> alphas;
        std::vector<vector_type> betas;
        scaling.reserve(T);
        alphas.reserve(T);
        betas.reserve(T);

        // calculate forward and backward coefficients
        forward(initial_model, sequence, ranges::back_inserter(alphas), ranges::back_inserter(scaling));
        backward(initial_model,
                  sequence | ranges::view::reverse,
                   scaling | ranges::view::reverse, ranges::back_inserter(betas));

        // calculate matrix updates
        matrix_type A;
        matrix_type B;
        vector_type pi(initial_model.states());

        for (index_type i = 0; i < initial_model.states(); ++i)
          pi(i) = alphas[0](i)*betas[T-1](i) / scaling[0];
        std::tie(A, B) = detail::baum_welch::update_matrices(sequence, alphas, betas, scaling[T-1], initial_model);

        return hidden_markov_model<float>(A, B, pi);
      }

  }

}
}

#endif /* HMM_ALGORITHM_BAUM_WELCH_H_ */
