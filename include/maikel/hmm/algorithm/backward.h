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

#include <iostream>
#include <gsl_assert.h>

#ifndef HMM_ALGORITHM_BACKWARD_H_
#define HMM_ALGORITHM_BACKWARD_H_

namespace maikel { namespace hmm {

template <class I, class J, class T>
    class backward_range_fn {
      public:
        using model       = hidden_markov_model<T>;
        using row_vector  = typename model::row_vector;
        using matrix      = typename model::matrix;
        using symbol_type = typename std::iterator_traits<I>::value_type;

        backward_range_fn() = delete;

        backward_range_fn(I seq_it, I seq_end, J scaling_it, model const& hmm)
        :  hmm_{&hmm}, seq_it_{seq_it}, seq_end_{seq_end}, scaling_it_{scaling_it},
          beta_(hmm.states()), next_beta_(hmm.states())
        {
          if (seq_it != seq_end)
            initial_coefficients(*scaling_it);
        }

        class iterator
        : public boost::iterator_facade<
              iterator, row_vector, std::input_iterator_tag, row_vector const&
         > {
          public:
            iterator() = default;
          private:
            friend class backward_range_fn;
            friend class boost::iterator_core_access;

            iterator(backward_range_fn& parent)
            : parent_{parent ? &parent : nullptr} {}

            backward_range_fn* parent_ = nullptr;

            row_vector const& dereference() const
            {
              Expects(parent_ && *parent_);
              return parent_->beta_;
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
        I seq_it_, seq_end_;
        J scaling_it_;
        row_vector beta_;
        row_vector next_beta_;

        void initial_coefficients(T scaling) noexcept
        {
          Expects(beta_.size() == hmm_->states());
          beta_.fill(scaling);
        }

        void recursion_advance(symbol_type s, T scaling) noexcept
        {
          using size_type = typename model::size_type;
          matrix const& A = hmm_->transition_matrix();
          matrix const& B = hmm_->symbol_probabilities();
          next_beta_.swap(beta_);

          // check pre conditions
          size_type states = A.rows();
          Expects(A.cols() == states);
          Expects(B.rows() == states);
          Expects(next_beta_.size() == states);
          Expects(beta_.size() == states);
          size_type ob = gsl::narrow<size_type>(s);
          Expects(0 <= ob && ob < B.cols());

          // recursion formula
          for (size_type i = 0; i < states; ++i) {
            beta_(i) = 0.0;
            for (size_type j = 0; j < states; ++j)
              beta_(i) += A(i,j)*B(j,ob)*next_beta_(j);
            beta_(i) *= scaling;
          }
        }

        bool next()
        {
          Expects(*this);
          symbol_type s = *seq_it_++;
          ++scaling_it_;
          if (seq_it_ == seq_end_)
            return false;
          recursion_advance(s, *scaling_it_);
          return true;
        }
    };

  template <class I, class J, class T>
    backward_range_fn<I, J, T>
    backward(I begin, I end, J scaling, hidden_markov_model<T> const& hmm)
    {
      return {begin, end, scaling, hmm};
    }

} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_BACKWARD_H_ */
