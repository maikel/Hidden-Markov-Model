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

#include <memory>
#include <range/v3/core.hpp>
#include <gsl_assert.h>

#ifndef HMM_ALGORITHM_BACKWARD_H_
#define HMM_ALGORITHM_BACKWARD_H_

namespace maikel { namespace hmm {

  template <class I, class S, class J, class T>
    class backward_input_range {
      public:
        using model       = hidden_markov_model<T>;
        using symbol_type = typename I::value_type;
        using size_type   = typename model::size_type;
        using row_vector  = typename model::row_vector;
        using value_type  = typename std::pair<T, row_vector>;

        struct iterator;

        /** \brief Providing no initial backward coefficients consumes the first symbol
         */
        backward_input_range(I seq_begin, S seq_end,
            J scaling_begin, hidden_markov_model<T> const& hmm) noexcept
        : hmm_{hmm}, seq_iter_{seq_begin}, seq_end_{seq_end},
          scaling_iter_{scaling_begin}, beta_(hmm.states()), next_beta_(hmm.states())
        {
          initial_coefficients(*scaling_iter_);
        }

        backward_input_range(I seq_begin, S seq_end,
            J scaling_begin, hidden_markov_model<T> const& hmm, row_vector const& betaT) noexcept
        : hmm_{hmm}, seq_iter_{seq_begin}, seq_end_{seq_end}, beta_{betaT}, next_beta_(hmm.states())
        {
        }

        /*
         * provide standard iterator access. this defines being a range
         */
        inline iterator begin() noexcept
        {
          return {this};
        }
        inline iterator end() noexcept
        {
          return {this};
        }

      private:

        hidden_markov_model<T> const& hmm_;
        I seq_iter_;
        S seq_end_;
        J scaling_iter_;
        row_vector beta_;
        row_vector next_beta_;

        void initial_coefficients(T scaling) noexcept
        {
          Expects(beta_.size() == hmm_.states());
          beta_.fill(scaling);
        }

        void recursion_advance(symbol_type s, T scaling) noexcept
        {
          auto const& A = hmm_.transition_matrix();
          auto const& B = hmm_.symbol_probabilities();
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
              beta_(i) += A(i,j)*B(j,ob)*next_beta_(j)*scaling;
          }
        }
    };

  /** \brief the forward range iterator is an abstract pointer to the
   *         current coefficients calculated by the hmm algorithm.
   *
   *  Each time the iterator gets incremented it we apply the recursion
   *  formula on the current forward coefficients and change them.
   */
  template <class I, class S, class J, class T>
    struct backward_input_range<I,S,J,T>::iterator
        : public std::iterator<std::input_iterator_tag, std::pair<T, row_vector const&>> {
      backward_input_range<I,S,J,T>* parent_;

      iterator(backward_input_range<I,S,J,T>* parent): parent_{parent}
      {
      }

      /** \brief dereference the forward iterator to the current pair of
       *         scaling factors and coefficients.
       */
      row_vector const& operator*() const noexcept
      {
        return parent_->beta_;
      }

      /** \brief advancing the iterator means changing the iterable.
       *
       *  You shall not return!
       */
      iterator& operator++() noexcept
      {
        parent_->recursion_advance(*parent_->seq_iter_++, *parent_->scaling_iter_++);
        return *this;
      }

      std::shared_ptr<row_vector> operator++(int) noexcept
      {
        ++(*this);
        return std::make_shared( parent_->next_beta_ );
      }

      /** \brief just check underlying symbol sequence iterators.
       */
      inline bool operator==(iterator sentinel) const noexcept
      {
        return parent_->seq_iter_ == sentinel.parent_->seq_end_;
      }

      /** \brief implicit definition through equality
       */
      inline bool operator!=(iterator sentinel) const noexcept
      {
        return !(*this == sentinel);
      }
    };

  template <class I, class S, class J, class T>
    inline backward_input_range<I, S, J, T>
    backward(I begin, S end, J scaling, hidden_markov_model<T> const& hmm) noexcept
    {
      return { begin, end, scaling, hmm };
    }

  template <class Rng, class J, class T,
             class I = ranges::range_iterator_t<Rng>,
             class S = ranges::range_sentinel_t<Rng>>
    inline backward_input_range<I, S, J, T>
    backward(Rng&& sequence, J scaling, hidden_markov_model<T> const& hmm) noexcept
    {
      return { std::begin(sequence), std::end(sequence), scaling, hmm };
    }

} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_BACKWARD_H_ */
