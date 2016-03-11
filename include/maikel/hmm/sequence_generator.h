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

#ifndef HMM_SEQUENCE_GENERATOR_H_
#define HMM_SEQUENCE_GENERATOR_H_

#include <random>
#include "maikel/hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  namespace detail {
    template <class float_type>
      class sequence_generator {
        public:
          using hmm         = typename ::maikel::hmm::hidden_markov_model<float_type>;
          using index_type  = typename hmm::size_type;
          using state_type  = index_type;
          using symbol_type = index_type;
          using matrix      = typename hmm::matrix;
          using row_vector  = typename hmm::row_vector;

        private:
          // random device stuff
          std::default_random_engine m_engine;
          std::uniform_real_distribution<float_type> m_uniform {0, 1};
          // current context variables
          hmm const& m_hmm;
          state_type m_current_state;

        public:
          explicit sequence_generator(hmm const& hmm)
          : m_engine(std::random_device()()), m_hmm(hmm)
          {
            float_type X = m_uniform(m_engine);
            m_current_state = find_by_distribution(m_hmm.initial_distribution(), X);;
          }

          symbol_type operator()() noexcept
          {
            float_type X = m_uniform(m_engine);
            // get next symbol
            symbol_type symbol =
                find_by_distribution(m_hmm.symbol_probabilities(), m_current_state, X);
            // advance a state
            m_current_state =
                find_by_distribution(m_hmm.transition_matrix(), m_current_state, X);
            return symbol;
          }

        private:
          index_type
          find_by_distribution(row_vector const& dist, float_type X)
          noexcept
          {
            float_type P_fn = 0;
            index_type state = 0;
            index_type max = dist.size();
            while (state < max) {
              P_fn += dist(state);
              if (P_fn < X)
                ++state;
              else
                break;
            }
            return state;
          }

          index_type
          find_by_distribution(matrix const& dist, index_type row, float_type X)
          noexcept
          {
            Expects(row < dist.rows());
            float_type P_fn = 0;
            index_type state = 0;
            index_type cols = dist.cols();
            while (state < cols) {
              P_fn += dist(row, state);
              if (P_fn < X)
                ++state;
              else
                break;
            }
            return state;
          }
      };
  } // namespace detail

  template <class float_type>
    detail::sequence_generator<float_type>
    make_sequence_generator(hidden_markov_model<float_type> const& hmm)
    {
      return detail::sequence_generator<float_type>(hmm);
    }

} // namespace hmm
} // namespace maikel

#endif /* HMM_SEQUENCE_GENERATOR_H_ */
