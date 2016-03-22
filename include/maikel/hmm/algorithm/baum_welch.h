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

#include <cstddef>
#include "maikel/hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  namespace detail { namespace baum_welch {

    template <class SeqI, class AlphaI, class BetaI, class T>
    class update_matrices_fn {
      public:
        using matrix = typename hidden_markov_model<T>::matrix;
        using row_vector = typename hidden_markov_model<T>::row_vector;

        update_matrices_fn() = delete;
        update_matrices_fn(size_t states, size_t symbols)
        : states_{states}, symbols_{symbols},
          xi_(states, states), B_(states, symbols), gamma_(states), gamma_sum_(states) {}

        std::pair<matrix const&, matrix const&> operator()(
            SeqI seq_it, SeqI seq_end,
            AlphaI alphas, BetaI betas, T scaling, hidden_markov_model<T> const& hmm)
        {
          Expects(seq_it != seq_end);
          size_t t_max = std::distance(seq_it, seq_end);
          matrix const& A = hmm.transition_matrix();
          matrix const& B = hmm.symbol_probabilities();
          xi_.setZero();
          B_.setZero();
          gamma_sum_.setZero();
          for (size_t t = 0; t < t_max-1; ++t) {
            gamma_.setZero();
            for (size_t i = 0; i < states_; ++i)
              for (size_t j = 0; j < states_; ++j) {
                T xi_t = alphas[t](i)*A(i,j)*B(j,seq_it[t+1])*betas[t+1](j);
                xi_(i,j) += xi_t;
                gamma_(i) += xi_t;
              }
            for (size_t j = 0; j < states_; ++j) {
              B_(j,seq_it[t]) += gamma_(j);
              gamma_sum_(j) += gamma_(j);
            }
          }
          for (size_t j = 0; j < states_; ++j)
            for (size_t i = 0; i < states_; ++i)
              xi_(i,j) /= gamma_sum_(i);

          for (size_t i = 0; i < states_; ++i) {
            T entry = alphas[t_max-1](i)*betas[t_max-1](i) / scaling;
            B_(i,seq_it[t_max-1]) += entry;
            gamma_sum_(i) += entry;
          }
          for (size_t i = 0; i < states_; ++i)
            for (size_t k = 0; k < symbols_; ++k)
              B_(i,k) /= gamma_sum_(i);

          return { xi_, B_ };
        }

      private:
        size_t states_, symbols_;
        matrix xi_;
        matrix B_;
        row_vector gamma_;
        row_vector gamma_sum_;
    };
  }}

  template <class SeqI, class AlphaI, class BetaI, class T>
  detail::baum_welch::update_matrices_fn<SeqI, AlphaI, BetaI, T>
  update_matrices(size_t states, size_t symbols)
  {
    return detail::baum_welch::update_matrices_fn<SeqI, AlphaI, BetaI, T>(states, symbols);
  }

//      std::pair<Matrix, Matrix>
//      update_matrices(
//          SeqRng&&   sequence,
//          AlphaRng&& alphas,
//          BetaRng&&  betas,
//          Float scaling,
//          HiddenMarkovModel const& model)
//      {
//        using Index   = typename HiddenMarkovModel::index_type;
//        using Vector  = typename HiddenMarkovModel::vector_type;
//        using namespace ranges;
//        using namespace gsl;
//
//        Expects(!empty(sequence) && !empty(betas) && !empty(alphas));
//
//        Index states  = model.states();
//        Index symbols = model.symbols();
//        Matrix xi     = Matrix::Zero(states, states);
//        Matrix B      = Matrix::Zero(states, symbols);
//        Vector g_sum  = Vector::Zero(states);
//        Vector gamma(states);
//
//        auto seq_iterator   = begin(sequence);
//        auto alpha_iterator = begin(alphas);
//        auto beta_iterator  = begin(betas);
//        while (seq_iterator+1 != end(sequence) &&
//            alpha_iterator+1 != end(alphas) && beta_iterator+1 != end(betas)) {
//          Index ob      = narrow<Index>(*seq_iterator++);
//          Index ob_next = narrow<Index>(*seq_iterator);
//          Expects(0 <= ob      && ob      < symbols);
//          Expects(0 <= ob_next && ob_next < symbols);
//
//          gamma.setZero();
//          Vector alpha = *alpha_iterator++;
//          Vector beta   = *((beta_iterator++)+1);
//          Expects(alpha.size() == states);
//          Expects(beta.size()  == states);
//          for (Index i = 0; i < states; ++i)
//            for (Index j = 0; j < states; ++j) {
//              float xi_t = alpha(i)*model.A(i,j)*model.B(j,ob_next)*beta(j);
//              xi(i,j)  += xi_t;
//              gamma(i) += xi_t;
//            }
//          for (Index j = 0; j < states; ++j) {
//            B(j,ob) += gamma(j);
//            g_sum(j) += gamma(j);
//          }
//        }
//        for (Index j = 0; j < states; ++j)
//          for (Index i = 0; i < states; ++i)
//            xi(i,j) /= g_sum(i);
//        Ensures(maikel::rows_are_probability_arrays(xi));
//
//        Expects(seq_iterator   != end(sequence));
//        Expects(alpha_iterator != end(alphas));
//        Expects(beta_iterator  != end(betas));
//        Index ob = narrow<Index>(*seq_iterator++);
//        Vector alpha = *alpha_iterator++;
//        Vector beta  = *beta_iterator++;
//        for (Index i = 0; i < states; ++i) {
//          float entry = alpha(i)*beta(i) / scaling;
//          B(i,ob) += entry;
//          g_sum(i) += entry;
//        }
//        for (Index i = 0; i < states; ++i)
//          for (Index k = 0; k < symbols; ++k)
//            B(i,k) /= g_sum(i);
//        Ensures(maikel::rows_are_probability_arrays(B));
//        Ensures(seq_iterator == end(sequence));
//        Ensures(alpha_iterator == end(alphas));
//        Ensures(beta_iterator == end(betas));
//
//        return std::make_pair(xi, B);
//      }
//  } }
//
//  namespace naive {
//
//    /**
//     * naive implementation of the baum-welch algorithm. just save everything
//     * into std::vector's and do not buffer anything in files. Just see if it
//     * works.
//     */
//    template <class float_type, class SeqRng>
//      hidden_markov_model<float_type>
//      baum_welch(
//          hidden_markov_model<float_type> const& initial_model,
//          SeqRng sequence)
//      {
//        // typedefs
//        using vector_type = typename hidden_markov_model<float_type>::vector_type;
//        using matrix_type = typename hidden_markov_model<float_type>::matrix_type;
//        using index_type  = typename hidden_markov_model<float_type>::index_type;
//
//        size_t T = sequence.size();
//
//        // allocate memory
//        std::vector<float_type> scaling;
//        std::vector<vector_type> alphas;
//        std::vector<vector_type> betas;
//        scaling.reserve(T);
//        alphas.reserve(T);
//        betas.reserve(T);
//
//        // calculate forward and backward coefficients
//        forward(initial_model, sequence, ranges::back_inserter(alphas), ranges::back_inserter(scaling));
//        backward(initial_model,
//                  sequence | ranges::view::reverse,
//                   scaling | ranges::view::reverse, ranges::back_inserter(betas));
//
//        // calculate matrix updates
//        matrix_type A;
//        matrix_type B;
//        vector_type pi(initial_model.states());
//
//        for (index_type i = 0; i < initial_model.states(); ++i)
//          pi(i) = alphas[0](i)*betas[T-1](i) / scaling[0];
//        std::tie(A, B) = detail::baum_welch::update_matrices(sequence, alphas,
//            betas | ranges::view::reverse, scaling[T-1], initial_model);
//
//        return hidden_markov_model<float_type>(A, B, pi);
//      }
//
//  }

}
}

#endif /* HMM_ALGORITHM_BAUM_WELCH_H_ */
