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

#include <utility>
#include <range/v3/all.hpp>
#include <range/v3/utility/concepts.hpp>
#include <gsl_assert.h>
#include "maikel/function_profiler.h"
#include "maikel/hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  /** \brief the forward range is not a container but a lazy representation of
   *         the recursive forward algorithm for hidden markov models.
   */
  template <class I, class S, class T>
    class forward_iterable {
      public:
        using model       = typename hidden_markov_model<T>;
        using symbol_type = typename I::value_type;
        using size_type   = typename model::size_type;
        using row_vector  = typename model::row_vector;

        /** \brief the forward range iterator is an abstract pointer to the
         *         current coefficients calculated by the hmm algorithm.
         *
         *  Each time the iterator gets incremented it we apply the recursion
         *  formula on the current forward coefficients and change them.
         */
        struct const_iterator
            : public std::iterator<std::input_iterator_tag, std::pair<T, row_vector>> {
          forward_iterable* parent_;

          const_iterator(forward_iterable* parent): parent_(parent)
          {
          }

          /** \brief dereference the forward iterator to the current pair of
           *         scaling factors and coefficients.
           */
          std::pair<T, row_vector const&> operator*() const noexcept
          {
            return {parent_->scaling_, parent_->alpha_};
          }

          /** \brief advancing the iterator means changing the iterable.
           *
           *  You shall not return!
           */
          const_iterator& operator++() noexcept
          {
            ++(parent_->seq_iter_);
            if (!(parent_->seq_iter_ == parent_->seq_end_)) {
              parent_->recursion_advance();
            return *this;
          }

          bool operator==(const_iterator const& sentinel) const noexcept
          {
            return parent_->seq_iter_ == sentinel.parent_->seq_end_;
          }
          bool operator!=(const_iterator const& sentinel) const noexcept
          {
            return parent_->seq_iter_ != sentinel.parent_->seq_end_;
          }
        };

        forward_iterable(I seq_begin, S seq_end, hidden_markov_model<T> const& hmm)
        : hmm_{hmm}, seq_iter_{seq_begin}, seq_end_{seq_end}, alpha_(hmm.state()), prev_alpha_(hmm.state())
        {
          if (seq_begin != seq_end) {
            std::tie(scaling_, alpha_) = initial_forward_coefficient(*seq_iter_);
            prev_alpha_ = alpha_;
          }
        }

        const_iterator begin()
        {
          return {this};
        }

        const_iterator end()
        {
          return {this};
        }

      private:

        hidden_markov_model<T> const& hmm_;
        I seq_iter_;
        S seq_end_;
        row_vector alpha_;
        T          scaling_;
        row_vector prev_alpha_; // keeping the previous values here
                                // prevents reallocation in each recursion step

        std::pair<T, row_vector const&>
          initial_coefficients(symbol_type s) noexcept
          {
            using size_type = typename hidden_markov_model<T>::size_type;
            auto const& B  = hmm_.symbol_probabilities();
            auto const& pi = hmm_.initial_distribution();

            // check pre conditions
            size_type states = pi.size();
            size_type ob = gsl::narrow<size_type>(s);
            Expects(B.rows() == states);
            Expects(0 <= ob && ob < B.cols());
            Expects(alpha_.size() == states);

            row_vector alpha(states);
            scaling_ = 0;
            for (size_type i = 0; i < states; ++i) {
              alpha_(i) = pi(i)*B(i,ob);
              scaling_ += alpha_(i);
            }
            scaling_ = scaling_ ? 1/scaling_ : 0;
            alpha_ *= scaling_;

            // check post conditions
            Ensures((!scaling_ && almost_equal<T>(alpha_.sum(), 0.0)) ||
                    ( scaling_ && almost_equal<T>(alpha_.sum(), 1.0))    );
            return std::make_pair(scaling_, alpha_);
          }

          std::pair<T, row_vector const&> recursion_advance()
          {
            auto const& A = hmm_.transition_matrix();
            auto const& B = hmm_.symbol_probabilities();
            symbol_type s = *seq_iter_;

            // check pre conditions
            size_type states = A.rows();
            Expects(A.cols() == states);
            Expects(B.rows() == states);
            Expects(prev_alpha_.size() == states);
            size_type ob = gsl::narrow<size_type>(s);
            Expects(0 <= ob && ob < B.cols());

            // recursion formula
            T scaling { 0 };
            for (size_type j = 0; j < states; ++j) {
              alpha(j) = 0.0;
              for (size_type i = 0; i < states; ++i)
                alpha(j) += prev_alpha(i)*A(i,j);
              alpha(j) *= B(j, ob);
              scaling += alpha(j);
            }
            scaling = scaling ? 1/scaling : 0;
            alpha *= scaling;

            // post conditions
            Ensures((!scaling && almost_equal<T>(alpha_.sum(), 0.0)) ||
                    ( scaling && almost_equal<T>(alpha_.sum(), 1.0))    );
            return scaling;
          }
    };

  template <class I, class S, class T>
    forward_iterable<I, S, T>
    forward(I begin, S end, hidden_markov_model<T> const& hmm)
    {
      return forward_iterable<I, S, T>(begin, end, hmm);
    }

  template <class Rng, class T,
             class I = ranges::range_iterator_t<Rng>,
             class S = ranges::range_sentinel_t<Rng>>
    forward_iterable<I, S, T>
    forward(Rng&& sequence, hidden_markov_model<T> const& hmm)
    {
      return forward_iterable<I, S, T>(std::begin(sequence), std::end(sequence), hmm);
    }

//  }
} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_FORWARD_H_ */
