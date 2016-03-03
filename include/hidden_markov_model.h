#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <array>
#include <cstddef>
#include "hmm/algorithm.h"

namespace mnb { namespace hmm {

  struct not_probability_arrays: public std::runtime_error {
      not_probability_arrays(std::string arg): std::runtime_error(arg) {}
  };

  template <class _floatT,
    std::size_t _NumStates, std::size_t _NumSymbols, typename =
        typename std::enable_if<std::is_floating_point<_floatT>::value>::type>
  struct hidden_markov_model
  {
      using float_type = _floatT;

      hidden_markov_model(
          matrix<float_type,_NumStates,_NumStates> const& _A,
          matrix<float_type,_NumStates,_NumSymbols> const& _B,
          std::array<float_type,_NumStates> const& _pi)
      : A(_A), B(_B), pi(_pi) {
        if (!is_right_stochastic_matrix(A) ||
            !is_right_stochastic_matrix(B) ||
            !is_probability_array(pi))
          throw not_probability_arrays("Some inputs in constructor do not " \
              "have a stochastical property.");
      }
      virtual ~hidden_markov_model() = default;

      constexpr std::size_t states() const noexcept {
        return _NumStates;
      }
      constexpr std::size_t symbols() const noexcept {
        return _NumSymbols;
      }

      matrix<float_type,_NumStates,_NumStates> A;
      matrix<float_type,_NumStates,_NumSymbols> B;
      std::array<float_type,_NumStates> pi;
  };

} // namespace hmm
} // namespace mnb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
