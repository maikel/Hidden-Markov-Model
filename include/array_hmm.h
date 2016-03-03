#ifndef ARRAY_HMM_H_
#define ARRAY_HMM_H_

#include <array>
#include <algorithm>
#include <exception>
#include <sstream>
#include <istream>

#include "gsl_util.h"

namespace nmb {



template<typename _tp>
class number_iterator: public std::iterator<std::input_iterator_tag, _tp> {
  public:
    number_iterator(_tp n) :
        current_number(n)
    {
    }
    _tp operator*() const noexcept
    {
      return current_number;
    }
    void operator++() noexcept
    {
      ++current_number;
    }
    void operator--() noexcept
    {
      --current_number;
    }
    bool operator==(const number_iterator& it) const noexcept
    {
      return it.current_number == current_number;
    }
  private:
    _tp current_number;
};

template<size_t N, size_t M, typename _Probtp = float>
class array_hmm {
  public:
    static constexpr size_t num_hidden_states = N;
    static constexpr size_t num_symbols = M;

    using size_type = std::size_t;
    using state_type = size_type;
    using symbol_type = size_type;
    using probability_type = _Probtp;
    using transition_matrix_type = matrix_type<probability_type, N, N>;
    using symbols_matrix_type = matrix_type<probability_type, N, M>;
    using state_distribution_type = std::array<probability_type, N>;

    struct probability_error: public std::runtime_error {
        probability_error(std::string what) :
            std::runtime_error(what)
        {
        }
    };

    array_hmm(
        transition_matrix_type const& A,
        symbols_matrix_type const& B,
        state_distribution_type const& initial) :
        m_A(A), m_B(B), m_pi(initial)
    {
      // check rows of transition matrix
      if (!is_right_stochastic_matrix(m_A) ||
      // check rows of symbol probability
          !is_right_stochastic_matrix(m_B) ||
          // check initial distribution
          !is_probability_array(m_pi))
        throw probability_error("Error while constructing a HMM.");
    }

    virtual ~array_hmm() = default;

    /**
     * Return the current probability to transition to hidden state `j` if
     * being in state `i`.
     */
    inline probability_type transition_probability(state_type i,
        state_type j) const noexcept
    {
      return m_A[i][j];
    }

    /**
     * Return the observation probability of symbol `k` under the restriction
     * of being in the hidden state `i`.
     */
    inline probability_type symbol_probability(state_type i,
        symbol_type k) const noexcept
    {
      return m_B[i][k];
    }

    /**
     * Return the probability to start in the hidden state `i`.
     */
    inline probability_type initial_probability(state_type i) const noexcept
    {
      return m_pi[i];
    }

    struct state_iterator: public number_iterator<state_type> {
        state_iterator(state_type n) :
            number_iterator<state_type>(n)
        {
        }
    };
    struct symbol_iterator: public number_iterator<symbol_type> {
        symbol_iterator(symbol_type m) :
            number_iterator<state_type>(m)
        {
        }
    };

    constexpr std::size_t num_states() const noexcept { return N; }
    constexpr std::size_t num_symbol() const noexcept { return M; }

    state_iterator begin_states() const noexcept
    {
      return state_iterator(0);
    }
    state_iterator end_states() const noexcept
    {
      return state_iterator(N);
    }
    symbol_iterator begin_symbols() const noexcept
    {
      return symbol_iterator(0);
    }
    symbol_iterator end_symbols() const noexcept
    {
      return symbol_iterator(M);
    }

  private:
    // Hidden Markov Model variables
    transition_matrix_type m_A;
    symbols_matrix_type m_B;
    state_distribution_type m_pi;
};

}

#endif /* ARRAY_HMM_H_ */
