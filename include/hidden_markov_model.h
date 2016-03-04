#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <array>
#include <cstddef>
#include <cassert>

#include "hmm/algorithm.h"

namespace mnb { namespace hmm {

  struct not_probability_arrays: public std::runtime_error {
      not_probability_arrays(std::string const& arg)
          : std::runtime_error(arg) {}
  };

  struct dimensions_not_consistent: public std::runtime_error {
      dimensions_not_consistent(std::string const& arg)
          : std::runtime_error(arg) {}
  };

  namespace array {

  template <class _floatT,
    std::size_t _NumStates, std::size_t _NumSymbols, typename =
        typename std::enable_if<std::is_floating_point<_floatT>::value>::type>
  struct hidden_markov_model
  {
      using float_type = _floatT;
      using transition_matrix_type = matrix<float_type,_NumStates,_NumStates>;
      using symbols_matrix_type = matrix<float_type,_NumStates,_NumSymbols>;
      using array_type = std::array<float_type,_NumStates>;

      hidden_markov_model(
          transition_matrix_type const& _A,
          symbols_matrix_type const& _B,
          array_type const& _pi)
      : A(_A), B(_B), pi(_pi) {
        if (!is_right_stochastic_matrix(A) ||
            !is_right_stochastic_matrix(B) ||
            !is_probability_array(pi))
          throw not_probability_arrays("Some inputs in constructor do not " \
              "have a stochastical property.");
      }
      virtual ~hidden_markov_model() = default;

      static constexpr std::size_t states() noexcept {
        return _NumStates;
      }
      static constexpr std::size_t symbols() noexcept {
        return _NumSymbols;
      }

      inline transition_matrix_type const&
      transition_matrix() const noexcept { return A; }
      inline symbols_matrix_type const&
      symbol_probabilities() const noexcept { return B; }
      inline array_type const&
      initial_distribution() const noexcept { return pi; }

      template <class InputIter, class OutputIter>
      inline void forward(InputIter start, InputIter end, OutputIter out)
      {
        ::mnb::hmm::forward(start, end, out, *this);
      }

      transition_matrix_type A;
      symbols_matrix_type B;
      array_type pi;
  };

  }

  namespace vector {

  template <class _floatT, typename =
        typename std::enable_if<std::is_floating_point<_floatT>::value>::type>
  struct hidden_markov_model
  {
      using float_type = _floatT;
      using transition_matrix_type = matrix<float_type>;
      using symbols_matrix_type = matrix<float_type>;
      using array_type = std::vector<float_type>;

      hidden_markov_model(
          transition_matrix_type const& _A,
          symbols_matrix_type const& _B,
          array_type const& _pi)
      : A(_A), B(_B), pi(_pi), _NumStates(A.size()), _NumSymbols(B[0].size()) {
        if (!is_right_stochastic_matrix(A) ||
            !is_right_stochastic_matrix(B) ||
            !is_probability_array(pi))
          throw not_probability_arrays("Some inputs in constructor do not " \
              "have a stochastical property.");
        if (A.size() != B.size() || A.size() != pi.size())
          throw dimensions_not_consistent("Dimensions of input matrices are " \
              "not consistent with each other.");
      }
      virtual ~hidden_markov_model() = default;

      inline std::size_t states() const noexcept {
        return _NumStates;
      }
      inline constexpr std::size_t symbols() const noexcept {
        return _NumSymbols;
      }

      inline transition_matrix_type const&
      transition_matrix() const noexcept { return A; }
      inline symbols_matrix_type const&
      symbol_probabilities() const noexcept { return B; }
      inline array_type const&
      initial_distribution() const noexcept { return pi; }

      template <class InputIter, class OutputIter>
      inline void forward(InputIter start, InputIter end, OutputIter out)
      {
        ::mnb::hmm::forward(start, end, out, *this);
      }

      transition_matrix_type A;
      symbols_matrix_type B;
      array_type pi;
      std::size_t _NumStates;
      std::size_t _NumSymbols;
  };

  }

  namespace detail {
    template <class hidden_markov_model,
              class Float = typename hidden_markov_model::float_type>
    class sequence_generator {
      public:
        sequence_generator(hidden_markov_model const& hmm)
        noexcept: m_engine(std::random_device()()), m_hmm(hmm)
        {
          Float X = uniform(m_engine);
          auto it = find_by_distribution(hmm.pi.begin(), hmm.pi.end(), X);
          m_current_state = std::distance(hmm.pi.begin(), it);
        }

        std::size_t operator()() noexcept
        {
          Float X = uniform(m_engine);

          // get next symbol
          auto symbol_it = find_by_distribution(
              m_hmm.B[m_current_state].begin(),
              m_hmm.B[m_current_state].end(), X);
          std::size_t symbol =
              std::distance(m_hmm.B[m_current_state].begin(), symbol_it);

          // advance a state
          auto state_it = find_by_distribution(
              m_hmm.A[m_current_state].begin(),
              m_hmm.A[m_current_state].end(), X);
          m_current_state =
              std::distance(m_hmm.A[m_current_state].begin(), state_it);

          return symbol;
        }

      private:
        // random device stuff
        std::default_random_engine m_engine;
        std::uniform_real_distribution<Float> uniform {0, 1};
        // current context variables
        const hidden_markov_model& m_hmm;
        std::size_t m_current_state;
    };
  }

  template <class hidden_markov_model,
            class Float = typename hidden_markov_model::float_type>
  detail::sequence_generator<hidden_markov_model> make_generator(
      hidden_markov_model const& hmm)
  {
      return detail::sequence_generator<hidden_markov_model>(hmm);
  }

} // namespace hmm
} // namespace mnb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
