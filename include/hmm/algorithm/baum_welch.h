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

  namespace naive {

    /**
     * naive implementation of the baum-welch algorithm. just save everything
     * into std::vector's and do not buffer anything in files. Just see if it
     * works.
     */
    template <class float_type,
               class SeqI, class SeqS>
      hidden_markov_model<float_type>
      baum_welch(
          hidden_markov_model<float_type> const& initial_model,
          SeqI sequence_iterator, SeqS sequence_sentinel)
      {
        // typedefs
        using vector_type = typename hidden_markov_model<float_type>::vector_type;
        using matrix_type = typename hidden_markov_model<float_type>::matrix_type;
        using difference_type = typename SeqI::difference_type;
        using index_type = typename hidden_markov_model<float_type>::index_type;

        // allocate memory
        std::vector<float> scaling;
        std::vector<vector_type> alphas;
        std::vector<vector_type> betas;
        difference_type sequence_length = std::distance(sequence_iterator, sequence_sentinel);
        scaling.reserve(sequence_length);
        alphas.reserve(sequence_length);
        betas.reserve(sequence_length);

        auto sequence_range = ranges::make_iterator_range(sequence_iterator, sequence_sentinel);
        // calculate forward and backward coefficients
        forward(initial_model, sequence_range, std::back_inserter(alphas), std::back_inserter(scaling));
        backward(initial_model,
            sequence_range | ranges::view::reverse,
                   scaling | ranges::view::reverse, ranges::back_inserter(betas));

        // calculate matrix updates
        matrix_type A = matrix_type::Zero(initial_model.states(), initial_model.states());
        matrix_type B = matrix_type::Zero(initial_model.states(), initial_model.symbols());
        vector_type pi(initial_model.states());

        vector_type gamma_sum = vector_type::Zero(initial_model.states());
        vector_type gamma;
        std::size_t t = 0;
        std::size_t T = gsl::narrow<std::size_t>(sequence_length);

        for (index_type i = 0; i < initial_model.states(); ++i)
          pi(i) = alphas[0](i)*betas[T-1](i) / scaling[0];
        while (sequence_iterator+1 != sequence_sentinel && t+1 < T) {
          index_type ob = gsl::narrow<index_type>(*sequence_iterator);
          index_type ob_next = gsl::narrow<index_type>(*(sequence_iterator+1));
          gamma = vector_type::Zero(initial_model.states());
          for (index_type j = 0; j < initial_model.states(); ++j) {
            for (index_type i = 0; i < initial_model.states(); ++i) {
              float entry = alphas[t](i)*initial_model.A(i,j)*initial_model.B(j,ob_next)*betas[T-t-1](j);
              A(i,j) += entry;
              gamma(i) += entry;
            }
          }
          for (index_type j = 0; j < initial_model.states(); ++j) {
            B(j,ob) += gamma(j);
            gamma_sum(j) += gamma(j);
          }
          ++sequence_iterator;
          ++t;
        }
        for (index_type j = 0; j < initial_model.states(); ++j)
          for (index_type i = 0; i < initial_model.states(); ++i)
            A(i,j) /= gamma_sum(i);

        index_type ob = gsl::narrow<index_type>(*sequence_iterator);
        for (index_type i = 0; i < initial_model.states(); ++i) {
          float entry = alphas[T-1](i)*betas[0](i) / scaling[T-1];
          B(i,ob) += entry;
          gamma_sum(i) += entry;
        }

        for (index_type i = 0; i < initial_model.states(); ++i)
          for (index_type k = 0; k < initial_model.symbols(); ++k)
            B(i,k) /= gamma_sum(i);

        return hidden_markov_model<float>(A, B, pi);
      }

  }

}
}

#endif /* HMM_ALGORITHM_BAUM_WELCH_H_ */
