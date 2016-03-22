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
#include <boost/iterator/iterator_facade.hpp>
#include <gsl_assert.h>

#include "maikel/hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  template <class InputIter, class T>
    class forward_range_fn {
      public:
        using model       = hidden_markov_model<T>;
        using row_vector  = typename model::row_vector;
        using matrix      = typename model::matrix;
        using symbol_type = typename std::iterator_traits<InputIter>::value_type;

        forward_range_fn() = delete;

        forward_range_fn(InputIter seq_it, InputIter seq_end, model const& hmm)
        :  hmm_{&hmm}, seq_it_{seq_it}, seq_end_{seq_end},
          alpha_{0,row_vector(hmm.states())}, prev_alpha_(hmm.states())
        {
          if (seq_it != seq_end)
            initial_coefficients(*seq_it);
        }

        class iterator
        : public boost::iterator_facade<
              iterator, std::pair<T, row_vector>, std::input_iterator_tag,
              std::pair<T, row_vector> const&
         > {
          public:
            iterator() = default;
          private:
            friend class forward_range_fn;
            friend class boost::iterator_core_access;

            iterator(forward_range_fn& parent)
            : parent_{parent ? &parent : nullptr} {}

            forward_range_fn* parent_ = nullptr;

            std::pair<T, row_vector> const& dereference() const
            {
              Expects(parent_ && *parent_);
              return parent_->alpha_;
            }

            void increment()
            {
              Expects(parent_ && *parent_);
              if (!parent_->next())
                parent_ = nullptr;
            }

            bool equal(iterator other) const noexcept
            {
              return parent_ == other.parent_;
            }
        };

        operator bool() const noexcept
        {
          return seq_it_ != seq_end_;
        }

        iterator begin() noexcept
        {
          return {*this};
        }

        iterator end() noexcept
        {
          return {};
        }

      private:
        model const* hmm_; // not owning
        InputIter seq_it_, seq_end_;
        std::pair<T, row_vector> alpha_;
        row_vector prev_alpha_;

        void initial_coefficients(symbol_type s)
        {
          using size_type = typename hidden_markov_model<T>::size_type;
          matrix const& B  = hmm_->symbol_probabilities();
          row_vector const& pi = hmm_->initial_distribution();

          // check pre conditions
          size_type states = pi.size();
          size_type ob = gsl::narrow<size_type>(s);
          Expects(B.rows() == states);
          Expects(0 <= ob && ob < B.cols());
          Expects(alpha_.second.size() == states);

          // initial formula
          T& scaling = alpha_.first;
          row_vector& alpha = alpha_.second;
          scaling = 0.0;
          for (size_type i = 0; i < states; ++i) {
            alpha(i) = pi(i)*B(i,ob);
            scaling += alpha(i);
          }
          scaling = scaling ? 1/scaling : 0;
          alpha *= scaling;

          // check post conditions
          Ensures((!scaling && almost_equal<T>(alpha.sum(), 0.0)) ||
                  ( scaling && almost_equal<T>(alpha.sum(), 1.0))    );
        }

        void recursion_advance(symbol_type s)
        {
          using size_type = typename hidden_markov_model<T>::size_type;
          matrix const& A = hmm_->transition_matrix();
          matrix const& B = hmm_->symbol_probabilities();
          prev_alpha_.swap(alpha_.second);

          // check pre conditions
          size_type states = A.rows();
          Expects(A.cols() == states);
          Expects(B.rows() == states);
          Expects(prev_alpha_.size() == states);
          size_type ob = gsl::narrow<size_type>(s);
          Expects(0 <= ob && ob < B.cols());

          // recursion formula
          T& scaling = alpha_.first;
          row_vector& alpha = alpha_.second;
          scaling = 0.0;
          for (size_type j = 0; j < states; ++j) {
            alpha(j) = 0.0;
            for (size_type i = 0; i < states; ++i)
              alpha(j) += prev_alpha_(i)*A(i,j);
            alpha(j) *= B(j, ob);
            scaling += alpha(j);
          }
          scaling = scaling ? 1/scaling : 0;
          alpha *= scaling;

          // post conditions
          Ensures((!scaling && almost_equal<T>(alpha.sum(), 0.0)) ||
                  ( scaling && almost_equal<T>(alpha.sum(), 1.0))    );
        }

        bool next()
        {
          Expects(*this);
          ++seq_it_;
          if (seq_it_ == seq_end_)
            return false;
          recursion_advance(*seq_it_);
          return true;
        }
    };

  template <class InputIter, class T>
    forward_range_fn<InputIter, T>
  forward(InputIter begin, InputIter end, hidden_markov_model<T> const& hmm)
  {
    return {begin, end, hmm};
  }

} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_FORWARD_H_ */
