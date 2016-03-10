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

#include "maikel/hmm/stochastical_conditions.h"

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
              matrix A;
              matrix B;
              row_vector pi;
              arguments_not_probability_arrays(
                  const matrix& A_,
                  const matrix& B_,
                  const row_vector& pi_,
                  const std::string& a): hmm_errors(a), A{A_}, B{B_}, pi{pi_} {}
          };

          hidden_markov_model(
              const matrix&     transition_matrix,
              const matrix&     symbol_matrix,
              const row_vector& initial_dist )
           : A { transition_matrix },
             B { symbol_matrix     },
             pi{ initial_dist      },
             num_states  { B.rows() },
             num_symbols { B.cols() }
          {
            if (!rows_are_probability_arrays(A) || !rows_are_probability_arrays(B)
                || !is_probability_array(pi.array()))
              throw arguments_not_probability_arrays
                { A, B, pi, "Some inputs in constructor do not have the stochastical property." };
            if (A.rows() != A.cols() || A.rows() != B.rows() || A.rows() != pi.cols())
              throw dimensions_not_consistent
                { "Dimensions of input matrices are not consistent with each other." };
          }

          inline size_type states() const noexcept  { return num_states;  }
          inline size_type symbols() const noexcept { return num_symbols; }

          inline const matrix&     transition_matrix()    const noexcept { return A; }
          inline const matrix&     symbol_probabilities() const noexcept { return B; }
          inline const row_vector& initial_distribution() const noexcept { return pi; }

        private:
          matrix A;
          matrix B;
          row_vector pi;
          size_type num_states;
          size_type num_symbols;
      };

} // namespace hmm
} // namespace maikel

#endif /* HIDDEN_MARKOV_MODEL_H_ */
