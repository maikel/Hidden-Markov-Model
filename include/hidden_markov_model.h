#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <iostream>
#include <iterator>
#include <random>

namespace nmb {

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

template <class _HMM_Tp>
  class sequence_generator
  {
  public:
    using hmm_type = _HMM_Tp;
    using symbol_type = typename hmm_type::symbol_type;
    using state_type = typename hmm_type::state_type;
    using prob_type = typename hmm_type::probability_type;

    sequence_generator(const hmm_type& hmm, std::random_device& rd)
    : m_engine(rd()), m_hmm(hmm) {
      prob_type X = uniform(m_engine);
      prob_type P_init { 0.0 };
      auto state = m_hmm.begin_states();
      while (!(state == m_hmm.end_states())) {
        P_init += m_hmm.initial_probability(*state);
        if (P_init < X)
          ++state;
        else
          break;
      }
      if (state == m_hmm.end_states())
        --state;
      m_current_state = *state;
      std::cout << "inital state " << m_current_state << std::endl;
    }

    symbol_type operator()() {
      prob_type X = uniform(m_engine);
      std::cout << "X: " << X << std::endl;

      // advance a state
      prob_type P_state = 0;
      auto state = m_hmm.begin_states();
      while (!(state == m_hmm.end_states()) && P_state < X) {
        P_state += m_hmm.transition_probability(m_current_state, *state);
        ++state;
      }
      if (state == m_hmm.end_states()) --state;
      m_current_state = *state;
      std::cout << m_current_state << std::endl;

      // determine symbol
      prob_type P_symbol = 0;
      auto symbol = m_hmm.begin_symbols();
      while (!(symbol == m_hmm.end_symbols()) && P_symbol < X) {
        P_symbol += m_hmm.symbol_probability(m_current_state, *symbol);
        ++symbol;
      }
      if (symbol == m_hmm.end_symbols()) --symbol;
      return *symbol;
    }

  private:
    // random device stuff
    std::default_random_engine m_engine;
    std::uniform_real_distribution<prob_type> uniform{0,1};
    // current context variables
    const hmm_type& m_hmm;
    typename hmm_type::state_type m_current_state;
  };

} // namespace nmb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
