#ifndef ALG_RO_
#define ALG_RO_

#include <algorithm> // all_of
#include <sstream>   // istringstream
#include <iterator>  // iterator_traits
#include <exception> // runtime_error
#include <cmath>     // pow10

#include "gsl_util.h"

#include "hidden_markov_model.h"

namespace mnb { namespace hmm {

  template<typename _Tp, std::size_t N, std::size_t M>
  using matrix = std::array<std::array<_Tp, M>, N>;

  struct matrix_read_error: public std::runtime_error {
      matrix_read_error(std::string arg) :
          std::runtime_error(arg)
      {
      }
  };

  struct matrix_write_error: public std::runtime_error {
      matrix_write_error(std::string arg) :
          std::runtime_error(arg)
      {
      }
  };

}

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

}

template<typename _Tp, std::size_t N, std::size_t M>
std::istream& operator>>(std::istream& in,
    mnb::hmm::matrix<_Tp,N,M>& matrix)
{
  std::istringstream linestream;
  std::string line;
  if (!in)
    throw mnb::hmm::matrix_read_error("Can not read from an invalid stream.");
  for (std::size_t n = N; n > 0 && std::getline(in, line); --n) {
    linestream.str(line);
    linestream.clear();
    for (std::size_t k = 0; k < M; ++k)
      if (!(linestream >> matrix[N - n][k]))
        throw mnb::hmm::matrix_read_error(
            "Error while parsing the matrix line: " + line);
  }
  if (!in)
    throw mnb::hmm::matrix_read_error(
        "Some error happened while reading a line.");

  return in;
}

template<typename _Tp, std::size_t N, std::size_t M>
std::ostream& operator<<(std::ostream& out,
    mnb::hmm::matrix<_Tp,N,M> const& matrix)
{
  if (!out)
    throw mnb::hmm::matrix_write_error(
        "Can not wirte to an invalid output stream.");
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t m = 0; m < M; ++m)
      out << matrix[n][m] << " ";
    out << "\n";
  }
  return out;
}

namespace mnb { namespace hmm {

  template <class Float, std::size_t N, std::size_t M>
  class sequence_generator {
    public:

      sequence_generator(hidden_markov_model<Float,N,M> const& hmm)
      noexcept: m_engine(std::random_device()), m_hmm(hmm)
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

  template <class Float, std::size_t N, std::size_t M>
  sequence_generator make_generator(
      hidden_markov_model<Float,N,M> const& hmm)
  {
      return sequence_generator<Float,N,M>(hmm);
  }

  /**
   * @brief Checks if a given floating point value is nonnegative.
   */
  template<typename _Tp>
  inline bool is_nonnegative(_Tp value) noexcept
  {
    return !(value < 0.0);
  }

  /**
   * @brief Checks if a given array is nonnegative and normed to one.
   */
  template<typename _floatT, std::size_t N, std::size_t accuracy = 15>
  inline bool
  is_probability_array(std::array<_floatT, N> const& p) noexcept
  {
    const _floatT eps = pow10(-gsl::narrow<int>(accuracy));
    bool entries_are_nonnegative = std::all_of(
        begin(p), end(p), is_nonnegative<_floatT>);
    _floatT total_sum = std::accumulate(begin(p), end(p), 0.0);
    bool array_is_normed = std::abs(total_sum - 1.0) < eps;
    return entries_are_nonnegative && array_is_normed;
  }

  /**
   * @brief Checks if every entry nonnegative and every row is normed to one.
   */
  template<typename _floatT, std::size_t N, std::size_t M,
    std::size_t accuracy = 15>
  inline bool
  is_right_stochastic_matrix(matrix<_floatT,N,M> const &matrix) noexcept
  {
    return std::all_of(begin(matrix), end(matrix),
        is_probability_array<_floatT, M, accuracy>);
  }

  template<class InputIter, class Float>
  InputIter
  find_by_distribution(InputIter start, InputIter end, Float X)
  noexcept {
      Float P_fn { 0.0 };
    while (!(start == end)) {
      P_fn += *start;
      if (P_fn < X)
        ++start;
      else
        break;
    }
    return start;
  }

} // namespace hmm
} // namespace nmb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
