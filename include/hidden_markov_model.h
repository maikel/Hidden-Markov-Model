#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <iostream>
#include <iterator>
#include <random>

namespace nmb {

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

template<typename _Probtp>
class hidden_markov_model {
  public:
    using probability_type = _Probtp;
    using size_type = std::size_t;
    using state_type = size_type;
    using symbol_type = size_type;

    virtual ~hidden_markov_model() noexcept = default;

    /**
     * Return the current probability to transition to hidden state `j` if being
     * in state `i`.
     */
    virtual probability_type
    transition_probability(state_type i, state_type j) const = 0;

    /**
     * Return the observation probability of symbol `k` under the restriction of
     * being in the hidden state `i`.
     */
    virtual probability_type
    symbol_probability(state_type i, symbol_type k) const = 0;

    /**
     * Return the probability to start in the hidden state `i`.
     */
    virtual probability_type
    initial_probability(state_type i) const = 0;
};

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
      std::cout << "inital state " << m_current_state << std::endl;
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
  constexpr std::size_t M = _Hmm_Tp::num_symbols;

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
    (*out) = make_pair(scaling, alpha);
    std::swap(alpha, pred_alpha);
    ++out;
    ++obstart;
  }
}

}
// namespace nmb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
