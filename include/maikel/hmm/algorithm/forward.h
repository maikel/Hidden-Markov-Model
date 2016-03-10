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

#ifndef HMM_ALGORITHM_FORWARD_H_
#define HMM_ALGORITHM_FORWARD_H_

#include <pair>
#include <range/v3/utility/static_const.hpp>
#include <range/v3/core.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/tail.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <gsl_assert.h>
#include "maikel/function_profiler.h"
#include "maikel/hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  template <class F, class I, class S>
    class forward_coefficients {
      public:
        using model       = typename hidden_markov_model<F>;
        using matrix      = typename model::matrix;
        using row_vector  = typename model::row_vector;
        using size_type   = typename model::size_type;
        using symbol_type     = typename ranges::iterator_value_t<I>;
        using symbol_iterator = typename I;
        using symbol_sentinel = typename S;
        using value_type = typename std::pair<F, row_vector>;

        class iterator : public std::iterator<std::forward_iterator_tag, value_type> {
          public:
            model const& hmm_;
            symbol_iterator symbol_;
            symbol_sentinel const symbol_end_;
            std::pair<F, row_vector> scaled_alpha_;

            bool operator==(S end) { return symbol_ == symbol_end_; }
            bool operator!=(S end) { return symbol_ != symbol_end_; }
            value_type const& operator*() { return scaled_alpha_; }

            iterator& operator++()
            {
              scaled_alpha_ = recursion_formula(*symbol_++);
              return *this;
            }

            iterator operator++(int)
            {
              iterator tmp(*this);
              scaled_alpha = recursion_formula(*symbol++);
              return tmp;
            }

          private:
            std::pair<F,row_vector>
            recursion_formula(symbol_type s) noexcept
            {
              matrix const& A = hmm.transition_matrix();
              matrix const& B = hmm.symbol_probabilities();

              // check pre conditions
              size_type states = A.rows();
              Expects(A.cols() == states);
              Expects(B.rows() == states);
              Expects(alpha.size() == states);
              size_type ob = gsl::narrow<size_type>(s);
              Expects(0 <= ob && ob < B.cols());

              // recursion formula
              row_vector next_alpha(alpha.size());
              F scaling { 0 };
              for (size_type j = 0; j < states; ++j) {
                next_alpha(j) = 0.0;
                for (size_type i = 0; i < states; ++i)
                  next_alpha(j) += alpha(i)*A(i,j);
                next_alpha(j) *= B(j, ob);
                scaling += next_alpha(j);
              }
              scaling = scaling ? 1/scaling : 0;
              next_alpha *= scaling;

              // post conditions
              Ensures((!scaling && almost_equal<F>(next_alpha.sum(), 0.0)) ||
                      ( scaling && almost_equal<F>(next_alpha.sum(), 1.0))    );
              alpha.swap(next_alpha);
              return std::make_pair(scaling, alpha);
            }
        };

        model const& hmm_;
        symbol_iterator sbegin_;
        symbol_sentinel send_;

      private:
        std::pair<F, row_vector>
        initial_alpha(symbol_type s) noexcept
        {
          matrix const& B      = hmm_.symbol_probabilities();
          row_vector const& pi = hmm_.initial_distribution();

          // check pre conditions
          Expects(B.rows() == pi.size());
          size_type states = pi.size();
          size_type ob = gsl::narrow<size_type>(s);
          Expects(0 <= ob && ob < B.cols());

          row_vector alpha_0(states);
          F scaling { 0 };
          for (size_type i = 0; i < states; ++i) {
            alpha_0(i) = pi(i)*B(i,ob);
            scaling += alpha_0(i);
          }
          scaling = scaling ? 1/scaling : 0;
          alpha_0 *= scaling;

          // check post conditions
          Ensures((!scaling && almost_equal<F>(alpha_0.sum(), 0.0)) ||
                  ( scaling && almost_equal<F>(alpha_0.sum(), 1.0))    );
          alpha_.swap(alpha_0);
          return std::make_pair(scaling, alpha_);
        }


      };


//  }
} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_FORWARD_H_ */
