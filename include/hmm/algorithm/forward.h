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

#include <range/v3/core.hpp>
#include <gsl_assert.h>
#include "types.h"

namespace maikel { namespace hmm {

  namespace detail { namespace forward {

    /**
     * @brief Calculate initial forward coefficients for given model parameters
     * `B` and `pi`. Formula is given by Rabiners paper
     *
     *   alpha_0(i) = pi(i) * B(i, O_1)
     *
     * This functions also normalize the vector alpha_0 in L_1-norm and returns
     * its norm factor in a std::pair.
     */
    template <class Symbol, class Float>
        // requires Integral<Symbol> && FloatingPoint<Float>
      std::pair<Float, VectorX<Float>>
      initial_alpha(Symbol s, const MatrixX<Float>& B, const VectorX<Float>& pi)
      noexcept
      {
        // check pre conditions
        using Index = typename MatrixX<Float>::Index;
        Expects(B.rows() == pi.size());
        Index states = pi.size();
        Index ob = gsl::narrow<Index>(s);
        Expects(0 <= ob && ob < B.cols());

        VectorX<Float> alpha_0(states);
        Float scaling {0};
        for (Index i = 0; i < states; ++i) {
          alpha_0(i) = pi(i)*B(i,ob);
          scaling += alpha_0(i);
        }
        scaling = scaling ? 1/scaling : 0;
        alpha_0 *= scaling;

        // check post conditions
        Ensures((!scaling && almost_equal<Float>(alpha_0.sum(), 0.0, 1)) ||
                ( scaling && almost_equal<Float>(alpha_0.sum(), 1.0, 1))    );
        return std::make_pair(scaling, alpha_0);
      }

    /**
     * @brief Calculate the next forward coeffients based on previous ones.
     * Coefficients are calculated based on the transition matrix `A` and
     * symbol probabilities `B`. The formula is given by (see Rabiner's paper)
     *
     *   alpha_{t+1}(j) = [ \sum_{i=1}^N alpha_{t}(i) A_{i,j} ] B_j(O_{t+1}).
     *
     * This function also normalize in L_1-norm and returns the factor.
     */
    template <class Symbol, class Float>
      Float
      recursion_formula(Symbol s, const MatrixX<Float>& A, const MatrixX<Float>& B,
          const VectorX<Float>& prev_alpha, VectorX<Float>& alpha)
      noexcept
      {
        // pre conditions
        using Index = typename MatrixX<Float>::Index;
        Index states = A.rows();
        Expects(A.cols() == states);
        Expects(B.rows() == states);
        Expects(prev_alpha.size() == states);
        Index ob = gsl::narrow<Index>(s);
        Expects(0 <= ob && ob < B.cols());

        // recursion formula
        Float scaling { 0 };
        for (Index j = 0; j < states; ++j) {
          alpha(j) = 0.0;
          for (Index i = 0; i < states; ++i)
            alpha(j) += prev_alpha(i)*A(i,j);
          alpha(j) *= B(j, ob);
          scaling += alpha(j);
        }
        scaling = scaling ? 1/scaling : 0;
        alpha *= scaling;

        // post conditions
        Ensures((!scaling && almost_equal<Float>(alpha.sum(), 0.0, 1)) ||
                ( scaling && almost_equal<Float>(alpha.sum(), 1.0, 1))    );
        return scaling;
      }
  } } // namespace detail } namespace forward }

  template <class SeqI, class SeqS, class ScalingOut, class AlphasOut, class HiddenMarkovModel>
    void forward(
      const HiddenMarkovModel& hmm,
      SeqI seq_iterator, SeqS seq_sentinel,
      AlphasOut alphas_out, ScalingOut scaling_out)
    {
      if (seq_iterator == seq_sentinel)
        return;
      using float_type = typename HiddenMarkovModel::float_type;
      float_type scaling_0;
      VectorX<float_type> alpha_0;
      std::tie(scaling_0, alpha_0) = detail::forward::initial_alpha(*seq_iterator, hmm.B, hmm.pi);
      Ensures(alpha_0.size() == hmm.states());
      *scaling_out++ = scaling_0;
      *alphas_out++  = alpha_0;
      ++seq_iterator;
      forward_with_initial_coefficients(hmm, seq_iterator, seq_sentinel, alphas_out, scaling_out, alpha_0);
    }

  template <class SeqRange, class ScalingOutIter, class AlphasOutIter, class HiddenMarkovModel>
    inline void forward(const HiddenMarkovModel& hmm,
        SeqRange&& sequence,
        AlphasOutIter alphas_out,
        ScalingOutIter scaling_out)
    {
      forward(hmm, ranges::begin(sequence), ranges::end(sequence), alphas_out, scaling_out);
    }

  template <class SeqI, class SeqS, class ScalingOut, class AlphasOut, class HiddenMarkovModel, class Float>
    void forward_with_initial_coefficients(
        const HiddenMarkovModel& hmm,
        SeqI seq_iterator, SeqS seq_sentinel,
        AlphasOut alphas_out, ScalingOut scaling_out,
        VectorX<Float> prev_alpha)
    {
      if (seq_iterator == seq_sentinel)
        return;
      using float_type = typename HiddenMarkovModel::float_type;
      using symbol_type = typename SeqI::value_type;
      Expects(prev_alpha.size() == hmm.states());
      float_type scaling;
      VectorX<float_type> alpha(hmm.states());

      while (seq_iterator != seq_sentinel) {
        symbol_type ob = *seq_iterator++;
        scaling = detail::forward::recursion_formula(ob, hmm.A, hmm.B, prev_alpha, alpha);
        *scaling_out++ = scaling;
        *alphas_out++  = alpha;
        alpha.swap(prev_alpha);
      }
    }

} // namespace hmm
} // namespace maikel

#endif /* HMM_ALGORITHM_FORWARD_H_ */
