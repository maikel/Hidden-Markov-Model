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
//#include <gsl.h>
#include "maikel/function_profiler.h"
#include "maikel/hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  namespace detail {
    template <class T, class S,
               class RowVector = typename hidden_markov_model<T>::row_vector>
      std::pair<T, RowVector>
      initial_forward_coefficient(S s, hidden_markov_model<T> const& hmm) noexcept
      {
        using size_type = typename hidden_markov_model<T>::size_type;
        auto const& B  = hmm.symbol_probabilities();
        auto const& pi = hmm.initial_distribution();

        // check pre conditions
        Expects(B.rows() == pi.size());
        size_type states = pi.size();
        size_type ob = gsl::narrow<size_type>(s);
        Expects(0 <= ob && ob < B.cols());

        RowVector alpha(states);
        T scaling { 0 };
        for (size_type i = 0; i < states; ++i) {
          alpha(i) = pi(i)*B(i,ob);
          scaling += alpha(i);
        }
        scaling = scaling ? 1/scaling : 0;
        alpha *= scaling;

        // check post conditions
        Ensures((!scaling && almost_equal<T>(alpha.sum(), 0.0)) ||
                ( scaling && almost_equal<T>(alpha.sum(), 1.0))    );
        return std::make_pair(scaling, alpha);
      }

    template <class T, class S,
               class RowVector = typename hidden_markov_model<T>::row_vector>
      T recursion_formula_forward_coefficients(
          S s,                          // input
          RowVector& alpha,             // output
          RowVector const& prev_alpha, // input
          hidden_markov_model<T> const& hmm) noexcept // input
      {
        using size_type = typename hidden_markov_model<T>::size_type;
        auto const& A = hmm.transition_matrix();
        auto const& B = hmm.symbol_probabilities();

        // check pre conditions
        size_type states = A.rows();
        Expects(A.cols() == states);
        Expects(B.rows() == states);
        Expects(prev_alpha.size() == states);
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
        Ensures((!scaling && almost_equal<T>(alpha.sum(), 0.0)) ||
                ( scaling && almost_equal<T>(alpha.sum(), 1.0))    );
        return scaling;
      }
  }

  /** \brief the forward range is not a container but a lazy representation of
   *         the recursive forward algorithm for hidden markov models.
   */
  template <class I, class S, class T>
    class forward_iterable {
      public:
        using row_vector  = typename hidden_markov_model<T>::row_vector;
        using size_type   = typename hidden_markov_model<T>::size_type;

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
            parent_->scaling_ = maikel::hmm::detail::recursion_formula_forward_coefficients(
                *(parent_->seq_iter_), parent_->prev_alpha_, parent_->alpha_, parent_->hmm_);
            ++parent_->seq_iter_;
            parent_->alpha_.swap(parent_->prev_alpha_);
            return *this;
          }

          bool operator==(S const& sentinel) const noexcept
          {
            return parent_->seq_iter_ == sentinel;
          }
          bool operator!=(S const& sentinel) const noexcept
          {
            return parent_->seq_iter_ != sentinel;
          }
        };

        forward_iterable(I seq_begin, S seq_end, hidden_markov_model<T> const& hmm)
        : hmm_{hmm}, seq_iter_{seq_begin}, seq_end_{seq_end}
        {
          if (seq_begin != seq_end) {
            std::tie(scaling_, alpha_) = detail::initial_forward_coefficient(*seq_iter_, hmm_);
            ++seq_iter_;
            prev_alpha_ = alpha_;
          }
        }

        const_iterator begin()
        {
          return {this};
        }

        S end()
        {
          return seq_end_;
        }

      private:

        hidden_markov_model<T> const& hmm_;
        I seq_iter_;
        S seq_end_;
        row_vector alpha_;
        T          scaling_;
        // keeping the previous values here prevents reallocation in each recursion step
        row_vector prev_alpha_;
//        T          prev_scaling
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
