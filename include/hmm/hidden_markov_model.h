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
#include <gsl_util.h>

#include "hmm/stochastical_conditions.h"

namespace maikel { namespace hmm {

  struct hmm_errors: public std::runtime_error {
    hmm_errors(std::string s): std::runtime_error(s) {}
  };

  struct dimensions_not_consistent: public hmm_errors {
    dimensions_not_consistent(const std::string& a): hmm_errors(a) {}
  };

  template <class T>
    class hidden_markov_model
      {
        public:
          using floating_point_type = T;
          using matrix     = typename Eigen::Matrix<floating_point_type, Eigen::Dynamic, Eigen::Dynamic>;
          using row_vector = typename Eigen::Matrix<floating_point_type, 1, Eigen::Dynamic>;
          using size_type  = typename matrix::Index;

          struct arguments_not_probability_arrays: public hmm_errors {
              matrix m_A;
              matrix m_B;
              row_vector m_pi;
              arguments_not_probability_arrays(
                  const matrix& A,
                  const matrix& B,
                  const row_vector& pi,
                  const std::string& a): hmm_errors(a), m_A{A}, m_B{B}, m_pi{pi} {}
          };

          hidden_markov_model(
              const matrix&     transition_matrix,
              const matrix&     symbol_matrix,
              const row_vector& initial_dist )
           : m_A { transition_matrix },
             m_B { symbol_matrix     },
             m_pi{ initial_dist      },
             m_num_states  { m_B.rows() },
             m_num_symbols { m_B.cols() }
          {
            if (!rows_are_probability_arrays(m_A) || !rows_are_probability_arrays(m_B)
                || !is_probability_array(m_pi.array()))
              throw arguments_not_probability_arrays
                { m_A, m_B, m_pi, "Some inputs in constructor do not have the stochastical property." };
            if (m_A.rows() != m_A.cols() || m_A.rows() != m_B.rows() || m_A.rows() != m_pi.cols())
              throw dimensions_not_consistent
                { "Dimensions of input matrices are not consistent with each other." };
          }

          inline size_type states() const noexcept  { return m_num_states;  }
          inline size_type symbols() const noexcept { return m_num_symbols; }

          inline const matrix&     transition_matrix()    const noexcept { return m_A; }
          inline const matrix&     symbol_probabilities() const noexcept { return m_B; }
          inline const row_vector& initial_distribution() const noexcept { return m_pi; }

        private:
          matrix m_A;
          matrix m_B;
          row_vector m_pi;
          size_type m_num_states;
          size_type m_num_symbols;
      };

} // namespace hmm
} // namespace maikel

#endif /* HIDDEN_MARKOV_MODEL_H_ */
