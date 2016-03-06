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

#include <range/v3/core.hpp>
#include <gsl_assert.h>
#include "types.h"

#ifndef HMM_ALGORITHM_BACKWARD_H_
#define HMM_ALGORITHM_BACKWARD_H_

namespace maikel { namespace hmm {

  namespace detail { namespace backward {

    template <class Float, class Index>
      inline VectorX<Float> initial_value(Float scaling, Index states) noexcept
      {
        Expects(scaling > 0);
        Expects(states > 0);
        return VectorX<Float>::Constant(states, scaling);
      }

    template <class Symbol, class Float>
      void recursion_formula(Symbol s, Float scaling,
          const MatrixX<Float>& A, const MatrixX<Float>& B,
          const VectorX<Float>& next_beta, VectorX<Float>& beta)
      noexcept
      {
        // pre conditions
        using Index = typename MatrixX<Float>::Index;
        Index states = A.rows();
        Expects(A.cols() == states);
        Expects(B.rows() == states);
        Expects(next_beta.size() == states);
        Expects(next_beta.size() == beta.size());
        Index ob = gsl::narrow<Index>(s);
        Expects(0 <= ob && ob < B.cols());

        // recursion formula
        for (Index i = 0; i < states; ++i) {
          beta(i) = 0.0;
          for (Index j = 0; j < states; ++j)
            beta(i) += A(i,j)*B(j,ob)*next_beta(j);
          beta(i) *= scaling;
        }
      }

  } }

  template <class SeqRange,
             class ScalingRange,
             class BetasOut,
             class HiddenMarkovModel>
    inline void backward(const HiddenMarkovModel& hmm,
        SeqRange&& sequence, ScalingRange&& scaling, BetasOut betas_out)
    {
      backward(hmm, ranges::begin(sequence), ranges::end(sequence),
                    ranges::begin(scaling),  ranges::end(scaling), betas_out);
    }

  template <class SeqI, class SeqS,
             class ScalingI, class ScalingS,
             class BetasOut,
             class HiddenMarkovModel>
    void backward(
        const HiddenMarkovModel& hmm,
        SeqI seq_iterator, SeqS seq_sentinel,
        ScalingI scaling_iterator, ScalingS scaling_sentinel,
        BetasOut betas_out)
    {
      if (seq_iterator == seq_sentinel || scaling_iterator == scaling_sentinel)
        return;
      using float_type = typename HiddenMarkovModel::float_type;

      // calculate initial beta_T
      float_type scaling = *scaling_iterator++;
      VectorX<float_type> beta = detail::backward::initial_value(scaling, hmm.states());
      Expects(beta.size() == hmm.states());

      // write beta_T to output iterator and advance iterators
      *betas_out++ = beta;
      backward_with_initial(hmm, seq_iterator, seq_sentinel,
          scaling_iterator, scaling_sentinel, betas_out, beta);
    }

  template <class SeqI, class SeqS,
             class ScalingI, class ScalingS,
             class BetasOut,
             class HiddenMarkovModel, class float_type>
    void backward_with_initial(
        const HiddenMarkovModel& hmm,
        SeqI seq_iterator, SeqS seq_sentinel,
        ScalingI scaling_iterator, ScalingS scaling_sentinel,
        BetasOut betas_out, VectorX<float_type> next_beta)
    {
      if (seq_iterator == seq_sentinel || scaling_iterator == scaling_sentinel)
        return;
      using Index = typename HiddenMarkovModel::index_type;
      using Symbol = typename SeqI::value_type;
      Index states = hmm.states();
      Expects(states > 0);
      VectorX<float_type> beta(states);
      while ((seq_iterator+1) != seq_sentinel &&
              scaling_iterator != scaling_sentinel) {
        Symbol ob          = *seq_iterator++;
        float_type scaling = *scaling_iterator++;
        detail::backward::recursion_formula(ob, scaling, hmm.A, hmm.B, next_beta, beta);
        *betas_out++ = beta;
        beta.swap(next_beta);
      }
    }

} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_BACKWARD_H_ */
