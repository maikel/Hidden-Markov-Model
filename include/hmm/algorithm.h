#ifndef ALG_RO_
#define ALG_RO_

#include <algorithm> // all_of
#include <sstream>   // istringstream
#include <iterator>  // iterator_traits
#include <exception> // runtime_error
#include <cmath>     // pow10
#include <istream>
#include <sstream>

#include "gsl_util.h"

namespace maikel { namespace hmm {

  template<class array_type,
            class float_type = typename array_type::Scalar,
            class index_type = typename array_type::Index>
    index_type
    find_by_distribution(array_type dist, float_type X) noexcept
    {
      float_type P_fn = 0;
      index_type state = 0;
      index_type max = dist.size();
      while (state < max) {
        P_fn += dist(state);
        if (P_fn < X)
          ++state;
        else
          break;
      }
      return state;
    }

  template <class InputIter, class OutputIter, class HMM,
      class transition_matrix = typename HMM::transition_matrix_type,
      class symbols_matrix = typename HMM::symbols_matrix_type,
      class array_type = typename HMM::array_type,
      class float_type = typename HMM::float_type,
      class symbol_type = typename InputIter::value_type>
  void forward_with_initial(
      InputIter start,
      InputIter end,
      OutputIter out,
      HMM const& hmm,
      array_type pred_alpha)
  {
    if (start == end)
      return;
    // get HMM properties
    transition_matrix const& A = hmm.transition_matrix();
    symbols_matrix const& B = hmm.symbol_probabilities();
    // do the recursion
    array_type alpha = make_array_like(pred_alpha);
    while (!(start == end)) {
      symbol_type ob = *start;
      assert(0 <= ob && ob < hmm.symbols());

      float_type scaling = 0.0;
      for (std::size_t j = 0; j < hmm.states(); ++j) {
        alpha(j) = 0.0;
        for (std::size_t i = 0; i < hmm.states(); ++i)
          alpha(i) += pred_alpha(i)*A(i,j);
        alpha(j) *= B(j,ob);
        scaling += alpha(j);
      }

      assert(scaling > 0);
      assert(is_almost_equal(
          scaling, std::accumulate(alpha.begin(), alpha.end(), 0.0f)));
      std::transform(alpha.begin(), alpha.end(), alpha.begin(),
          [scaling](float_type a){ return a / scaling; });
      *out = std::make_pair(1/scaling, alpha);
      std::swap(alpha, pred_alpha);
      ++out;
      ++start;
    }
  }

  template <
        class ObInputIter,
        class OutputIter,
        class HMM,
        class transition_matrix = typename HMM::transition_matrix_type,
        class symbols_matrix    = typename HMM::symbols_matrix_type,
        class array_type        = typename HMM::array_type,
        class float_type        = typename HMM::float_type,
        class symbol_type       = typename ObInputIter::value_type>
  void
  forward(
      ObInputIter ob_start,
      ObInputIter ob_end,
      OutputIter out,
      HMM const& hmm)
  {
    if (ob_start == ob_end)
      return;
    // get HMM properties
    symbols_matrix const& B = hmm.symbol_probabilities();
    array_type const& pi = hmm.initial_distribution();
    // determine initial alpha_0
    symbol_type ob = *ob_start;
    assert(0 <= ob && ob < hmm.symbols());
    array_type alpha = make_array_like(pi);

    float_type scaling{ 0.0 };
    for (std::size_t i=0; i < hmm.states(); ++i) {
      alpha(i) = pi(i)*B(i, ob);
      scaling += alpha(i);
    }

    assert(scaling > 0);
    assert(is_almost_equal(scaling, std::accumulate(alpha.begin(), alpha.end(), 0.0f)));
    std::transform(alpha.begin(), alpha.end(), alpha.begin(),
        [scaling](float_type a){ return a / scaling; });
    *out = std::make_pair(1/scaling, alpha);
    ++out;
    ++ob_start;

    // start the recursion formula with our initial alpha_0
    forward_with_initial(ob_start, ob_end, out, hmm, alpha);
  }

//  template <
//        class ObInputIter,
//        class ScalingInputIter,
//        class OutputIter,
//        class HMM,
//        class transition_matrix = typename HMM::transition_matrix_type,
//        class symbols_matrix    = typename HMM::symbols_matrix_type,
//        class array_type        = typename HMM::array_type,
//        class float_type        = typename HMM::float_type,
//        class symbol_type       = typename ObInputIter::value_type>
//  void
//  backward_with_initial(
//      ObInputIter ob_start,
//      ObInputIter ob_end,
//      ScalingInputIter scalit,
//      OutputIter out,
//      HMM const& hmm,
//      array_type next_beta)
//  {
//    if (ob_start == ob_end)
//      return;
//    // get HMM properties
//    transition_matrix const& A = hmm.transition_matrix();
//    symbols_matrix const& B = hmm.symbol_probabilities();
//    array_type beta = make_array_like(next_beta);
//    assert(beta.size() == hmm.states());
//    while (!(ob_start == ob_end)) {
//      symbol_type ob = *ob_start;
//      assert(0 <= ob && ob < hmm.symbols());
//      float_type scaling = *scalit;
//      assert(scaling > 0);
//      for (std::size_t i = 0; i < hmm.states(); ++i) {
//        beta[i] = 0.0;
//        for (std::size_t j = 0; j < hmm.states(); ++j)
//          beta[i] += A[i][j]*B[j][ob]*next_beta[j];
//        beta[i] *= scaling;
//      }
//      *out = beta;
//      std::swap(beta, next_beta);
//      ++out;
//      ++ob_start;
//      ++scalit;
//    }
//  }
//
//  template <
//        class ObInputIter,
//        class ScalingInputIter,
//        class OutputIter,
//        class HMM,
//        class transition_matrix  = typename HMM::transition_matrix_type,
//        class symbols_matrix     = typename HMM::symbols_matrix_type,
//        class array_type         = typename HMM::array_type,
//        class float_type         = typename HMM::float_type,
//        class symbol_type        = typename ObInputIter::value_type>
//  void
//  backward(
//      ObInputIter ob_start,
//      ObInputIter ob_end,
//      ScalingInputIter scalit,
//      OutputIter out,
//      HMM const& hmm)
//  {
//    if (ob_start == ob_end)
//      return;
//    // get HMM properties
//    symbols_matrix const& B = hmm.symbol_probabilities();
//    array_type const& pi = hmm.initial_distribution();
//    // determine initial beta_T
//    array_type beta = make_array_like(pi);
//    assert(beta.size() == hmm.states());
//    float_type scaling = *scalit;
//    assert(scaling > 0);
//    for (std::size_t i=0; i < hmm.states(); ++i) {
//      beta[i] = scaling;
//    }
//    *out = beta;
//    ++out;
//    ++scalit;
//    --ob_end;
//
//    // start the recursion formula with our initial beta_T
//    backward_with_initial(ob_start, ob_end, scalit, out, hmm, beta);
//  }


} // namespace hmm
} // namespace nmb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
