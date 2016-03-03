#ifndef HIDDEN_MARKOV_MODEL_H_
#define HIDDEN_MARKOV_MODEL_H_

#include <array>
#include <cstddef>
#include <cassert>

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

      static constexpr std::size_t states() noexcept {
        return _NumStates;
      }
      static constexpr std::size_t symbols() noexcept {
        return _NumSymbols;
      }

      template <class InputIter, class OutputIter>
      void forward(InputIter start, InputIter end, OutputIter out) const
      {
        if (start == end)
          return;

        constexpr std::size_t N = states();

        // determine alpha_0
        std::size_t ob = *start;
        assert(0 <= ob && ob < symbols());
        std::array<float_type, N> alpha;
        float_type scaling{ 0.0 };
        for (std::size_t i=0; i < N; ++i) {
          alpha[i] = pi[i]*B[i][ob];
          scaling += alpha[i];
        }
        assert(scaling > 0);
        assert(is_almost_equal(
            scaling, std::accumulate(alpha.begin(), alpha.end(), 0.0f)));
        std::transform(alpha.begin(), alpha.end(), alpha.begin(),
            [scaling](float_type a){ return a / scaling; });
        *out = std::make_pair(1/scaling, alpha);
        ++out;
        ++start;

        // do the recursion
        std::array<float_type, N> pred_alpha(alpha);
        while (!(start == end)) {
          ob = *start;
          assert(0 <= ob && ob < symbols());
          scaling = 0.0;
          for (std::size_t i = 0; i < N; ++i) {
            alpha[i] = 0.0;
            for (std::size_t j = 0; j < N; ++j)
              alpha[i] += pred_alpha[j]*A[j][i];
            alpha[i] *= B[i][ob];
            scaling += alpha[i];
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

      matrix<float_type,_NumStates,_NumStates> A;
      matrix<float_type,_NumStates,_NumSymbols> B;
      std::array<float_type,_NumStates> pi;
  };

  namespace detail {
    template <class Float, std::size_t N, std::size_t M>
    class sequence_generator {
      public:
        sequence_generator(hidden_markov_model<Float,N,M> const& hmm)
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
        const hidden_markov_model<Float,N,M>& m_hmm;
        std::size_t m_current_state;
    };
  }

  template <class Float, std::size_t N, std::size_t M>
  detail::sequence_generator<Float,N,M> make_generator(
      hidden_markov_model<Float,N,M> const& hmm)
  {
      return detail::sequence_generator<Float,N,M>(hmm);
  }

} // namespace hmm
} // namespace mnb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
