#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <iostream>
#include <iterator>
#include <random>
#include <cassert>

#include <boost/iterator/transform_iterator.hpp>

#include "array_hmm.h"

namespace nmb {

template<typename _Tp, std::size_t N, std::size_t M>
using matrix_type = std::array<std::array<_Tp, M>, N>;

template<class Iter, class Probability, class Functor>
// requires InputIterator<Iter> && FloatingPoint<Probability>
auto find_by_distribution(Iter start, Iter end, Probability X,
    Functor distribution) noexcept
{
  Probability P_fn { 0.0 };
  while (!(start == end)) {
    P_fn += distribution(*start);
    if (P_fn < X)
      ++start;
    else
      break;
  }
  return *start;
}

template<class _HMM_Tp>
class sequence_generator {
  public:
    using hmm_type = _HMM_Tp;
    using symbol_type = typename hmm_type::symbol_type;
    using state_type = typename hmm_type::state_type;
    using prob_type = typename hmm_type::probability_type;

    sequence_generator(const hmm_type& hmm, std::random_device& rd) noexcept:
    m_engine(rd()), m_hmm(hmm)
    {
      prob_type X = uniform(m_engine);
      prob_type P_init {0.0};
      m_current_state = find_by_distribution(hmm.begin_states(), hmm.end_states(), X,
          [this](auto p) {return m_hmm.initial_probability(p);});
    }

    symbol_type operator()() noexcept
    {
      prob_type X = uniform(m_engine);
      // advance a state
      m_current_state = find_by_distribution(m_hmm.begin_states(), m_hmm.end_states(), X, [this](auto p) {
            return m_hmm.transition_probability(m_current_state, p);
          });
      // determine symbol
      return find_by_distribution(m_hmm.begin_symbols(), m_hmm.end_symbols(), X, [this](auto p) {
            return m_hmm.symbol_probability(m_current_state, p);
          });
    }

    private:
    // random device stuff
    std::default_random_engine m_engine;
    std::uniform_real_distribution<prob_type> uniform {0, 1};
    // current context variables
    const hmm_type& m_hmm;
    typename hmm_type::state_type m_current_state;
  };


template <class InIter, class OutIter, class _Hmm_Tp>
void forward(InIter obstart, InIter obend, OutIter out, _Hmm_Tp const& hmm) noexcept
{
  if (obstart == obend)
    return;

  auto A = [&hmm](auto i, auto j){ return hmm.transition_probability(i,j); };
  auto B = [&hmm](auto i, auto k){ return hmm.symbol_probability(i,k); };
  auto pi = [&hmm](auto i) { return hmm.initial_probability(i); };
  using symbol_type = typename _Hmm_Tp::symbol_type;
  using probability_type = typename _Hmm_Tp::probability_type;

  constexpr std::size_t N = _Hmm_Tp::num_hidden_states;

  symbol_type symbol_start = *obstart;
  std::array<probability_type, N> alpha;
  probability_type scaling = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    alpha[i] = pi(i)*B(i, symbol_start);
    scaling += alpha[i];
  }
  std::transform(alpha.begin(), alpha.end(), alpha.begin(),
      [scaling](auto a) { return a/scaling; });
  (*out) = make_pair(scaling, alpha);
  ++out;
  ++obstart;

  std::array<probability_type, N> pred_alpha(alpha);
  while (!(obstart == obend)) {
    symbol_type ob = *obstart;
    scaling = 0.0;
    for (std::size_t j = 0; j < N; ++j) {
      alpha[j] = 0.0;
      for (std::size_t i = 0; i < N; ++i)
        alpha[j] += pred_alpha[i]*A(i,j);
      alpha[j] *= B(j, ob);
      scaling += alpha[j];
    }
    std::transform(alpha.begin(), alpha.end(), alpha.begin(),
        [scaling](auto a) {return a/scaling;});
    (*out) = make_pair(1/scaling, alpha);
    std::swap(alpha, pred_alpha);
    ++out;
    ++obstart;
  }
}

template <class InIter, class OutIter, class InIterScale, class _Hmm_Tp>
void backward(InIter obstart, InIter obend, InIterScale scalin,
    OutIter out, _Hmm_Tp const& hmm) noexcept
{
  if (obstart == obend)
    return;

  auto A = [&hmm](auto i, auto j){ return hmm.transition_probability(i,j); };
  auto B = [&hmm](auto i, auto k){ return hmm.symbol_probability(i,k); };
  using symbol_type = typename _Hmm_Tp::symbol_type;
  using probability_type = typename _Hmm_Tp::probability_type;

  constexpr std::size_t N = _Hmm_Tp::num_hidden_states;

  std::array<probability_type, N> beta;
  probability_type scaling = *scalin;
  ++scalin;
  for (std::size_t i = 0; i < N; ++i)
    beta[i] = scaling;
  *out = beta;
  ++out;
  ++obstart;
  ++scalin;

  std::array<probability_type, N> next_beta(beta);
  while (!(obstart == obend)) {
    symbol_type ob = *obstart;
    scaling = *scalin;
    for (std::size_t j = 0; j < N; ++j) {
      beta[j] = 0.0;
      for (std::size_t i = 0; i < N; ++i)
        beta[j] += A(i,j)*B(j, ob)*next_beta[i];
      beta[j] *= scaling;
    }
    *out = beta;
    std::swap(beta, next_beta);
    ++scalin;
    ++out;
    ++obstart;
  }
}

template <class AlphaIter, class BetaIter, class OutIter>
void gamma(AlphaIter alpha_start, AlphaIter alpha_end, BetaIter beta_start,
    OutIter output) noexcept {
  if (alpha_start == alpha_end)
    return;

  using probability_type = typename decltype(*alpha_start)::value_type;
  constexpr std::size_t N = alpha_start->second.size();

  std::array<probability_type, N> gamma;
  while (!(alpha_start == alpha_end)) {
    for (std::size_t i = 0; i < N; ++i)
      gamma[i] = (alpha_start->second)[i]*(*beta_start)[i] / alpha_start->first;
    *output = gamma;
    ++output;
    ++alpha_start;
    ++beta_start;
  }
}

template <class ObsIter, class AlphaIter, class BetaIter, class OutIter, class _Hmm_Tp>
void xi(ObsIter obsbegin, ObsIter obsend,
    AlphaIter alpha_start, BetaIter beta_start,
    OutIter output, _Hmm_Tp const& hmm) noexcept {
  if (obsbegin == obsend)
    return;

  auto A = [&hmm](auto i, auto j){ return hmm.transition_probability(i,j); };
  auto B = [&hmm](auto i, auto k){ return hmm.symbol_probability(i,k); };
  using symbol_type = typename decltype(*obsbegin)::value_type;
  using probability_type = typename decltype(*alpha_start)::value_type;
  constexpr std::size_t N = alpha_start->second.size();

  std::array<std::array<probability_type, N>, N> matrix;
  while (!(obsbegin == obsend)) {
    auto const& alpha = alpha_start->second;
    auto const& beta  = *beta_start;
    symbol_type ob = *obsbegin;
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j)
        matrix[i][j] = alpha[i] * A(i,j) * B(j, ob) * beta[j];
    *output = matrix;
    ++obsbegin;
    ++alpha_start;
    ++beta_start;
  }
}

template <class ObsIter, class _Hmm_Tp>
_Hmm_Tp baum_welch(ObsIter obsbegin, ObsIter obsend, _Hmm_Tp const& hmm) noexcept
{
  if (obsbegin == obsend)
    return hmm;

  using probability_type = typename _Hmm_Tp::probability_type;
  using symbol_type = typename _Hmm_Tp::symbol_type;
  constexpr std::size_t N = _Hmm_Tp::num_hidden_states;
  constexpr std::size_t M = _Hmm_Tp::num_symbols;

  std::vector<std::pair<probability_type,std::array<probability_type, N>>> sc_alphas;
  std::vector<std::array<probability_type, N>> betas;

  forward(obsbegin, obsend, std::back_inserter(sc_alphas), hmm);
  auto scaling_start = boost::make_transform_iterator(sc_alphas.rbegin(),
        [](std::pair<probability_type, std::array<probability_type, N>>
            const& pair) { return pair.first; });
  backward(obsbegin, obsend, scaling_start, std::back_inserter(betas), hmm);

  matrix_type<probability_type, N, N> xi_sum;
  matrix_type<probability_type, N, M> B_sum;
  std::array<probability_type, N> pi;
  std::array<probability_type, N> gamma_sum;

  auto sc_alpha = sc_alphas.begin();
  auto beta_next = betas.rbegin();
  ++beta_next;
  auto betait = betas.rbegin();
  auto obit = obsbegin;
  ++obit;

  /* Set initial distribution.
   */
  probability_type scaling = sc_alpha->first;
  std::array<probability_type, N>& alpha = sc_alpha->second;
  std::array<probability_type, N>& beta = *betait;
  for (std::size_t i = 0; i < N; ++i)
    pi[i] = alpha[i]*beta[i]/scaling;
  assert(is_probability_array(pi));

  symbol_type ob;
  while (!(obit == obsend)) {
    ob = *obit;
    std::array<probability_type, N> const& beta_n = *beta_next;
    scaling = sc_alpha->first;
    alpha = sc_alpha->second;
    beta = *betait;
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j)
        xi_sum[i][j] += alpha[i] * hmm.transition_probability(i,j) * hmm.symbol_probability(j, ob) * beta_n[j];
      gamma_sum[i] += alpha[i]*beta[i]/scaling;
      // do not check for next observation!
      --obit;
      for (std::size_t k = 0; k < M; ++k)
        if (ob == k)
          B_sum[i][k] += alpha[i]*beta[i]/scaling;
      ++obit;
    }
    ++betait;
    ++beta_next;
    ++sc_alpha;
    ++obit;
  }
  std::cerr << xi_sum << std::endl;
  // finalize xi_sum by norm
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      xi_sum[i][j] /= gamma_sum[i];
  std::cerr << xi_sum << std::endl;
  std::copy(gamma_sum.begin(), gamma_sum.end(), std::ostream_iterator<probability_type>(std::cout, " "));
  assert(is_right_stochastic_matrix(xi_sum));

  // add last element to gamma sum
  scaling = sc_alpha->first;
  alpha = sc_alpha->second;
  beta = *betait;
  --obit;
  ob = *obit;
  for (std::size_t i = 0; i < N; ++i) {
    gamma_sum[i] += alpha[i]*beta[i]/scaling;
    for (std::size_t k = 0; k < M; ++k) {
      if (ob == k)
        B_sum[i][k] += alpha[i]*beta[i]/scaling;
      B_sum[i][k] /= gamma_sum[i];
    }
  }

  return _Hmm_Tp(xi_sum, B_sum, pi);
}

} // namespace nmb


#endif /* HIDDEN_MARKOV_MODEL_H_ */
