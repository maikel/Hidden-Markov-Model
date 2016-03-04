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

namespace mnb { namespace hmm {

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

  namespace array {

  template<typename _Tp, std::size_t N, std::size_t M>
  using matrix = std::array<std::array<_Tp, M>, N>;

  }

  namespace vector {

    template<typename _Tp>
    using matrix = std::vector<std::vector<_Tp>>;

    namespace detail {

    inline std::istringstream &getline(std::istream& in, std::istringstream& linestream)
    {
      std::string line;
      std::getline(in, line);
      linestream.str(line);
      linestream.clear();
      return linestream;
    }

    }

    template <class T>
    matrix<T>
    read_matrix(std::istream& in, std::size_t rows, std::size_t cols)
    {
      Expects(rows > 0);
      Expects(cols > 0);
      matrix<T> mat(rows, std::vector<T>(cols));
      assert(mat.size() == rows);
      assert(std::all_of(mat.begin(), mat.end(),
          [](std::vector<T> const& row) { return row.size() == cols; }));
      std::istringstream line;
      for (std::size_t i = 0; i < rows; ++i) {
        detail::getline(in, line);
        for (std::size_t j = 0; j < cols; ++j)
          if (!(line >> mat[i][j]))
            throw matrix_read_error("error while reading matrix entries.");
      }
      return mat;
    }

    template <class T>
    std::vector<T>
    read_array(std::istream& in, std::size_t max_n)
    {
      Expects(max_n > 0);
      std::vector<T> v(max_n);
      std::istringstream line;
      detail::getline(in, line);
      assert(v.size() == max_n);
      for (std::size_t i = 0; i < max_n; ++i)
        if (!(line >> v[i]))
          throw matrix_read_error("error while reading vector entries.");
      return v;
    }

  }

}

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

}

template<typename _Tp, std::size_t N, std::size_t M>
std::istream& operator>>(std::istream& in,
    mnb::hmm::array::matrix<_Tp,N,M>& matrix)
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
    mnb::hmm::array::matrix<_Tp,N,M> const& matrix)
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

  template<class InputIter, class Float>
  InputIter
  find_by_distribution(InputIter start, InputIter end, Float X)
  noexcept {
      Float P_fn { 0.0f };
    while (!(start == end)) {
      P_fn += *start;
      if (P_fn < X)
        ++start;
      else
        break;
    }
    return start;
  }

  /**
   * @brief Checks if a given floating point value is nonnegative.
   */
  template<typename _Tp>
  inline bool is_nonnegative(_Tp value) noexcept
  {
    return !(value < 0.0f);
  }

  template <typename _floatT, std::size_t accuracy = 4>
  inline bool is_almost_equal(_floatT a, _floatT b)
  {
    const _floatT eps = pow10(-gsl::narrow<int>(accuracy));
    return (std::abs(a-b) < eps);
  }

  /**
   * @brief Checks if a given array is nonnegative and normed to one.
   */
  template<class _containerT, std::size_t accuracy = 4>
  inline bool
  is_probability_array(_containerT const& p)
  {
    using _floatT = typename _containerT::value_type;
    bool entries_are_nonnegative = std::all_of(
        begin(p), end(p), is_nonnegative<_floatT>);
    _floatT total_sum = std::accumulate(begin(p), end(p), 0.0f);
    bool array_is_normed = is_almost_equal<_floatT, accuracy>(total_sum, 1.0f);
    return entries_are_nonnegative && array_is_normed;
  }

  /**
   * @brief Checks if every entry nonnegative and every row is normed to one.
   */
  template<class _containerT, std::size_t accuracy = 4>
  inline bool
  is_right_stochastic_matrix(_containerT const& matrix)
  {
    return std::all_of(begin(matrix), end(matrix),
        is_probability_array<typename _containerT::value_type, accuracy>);
  }

  template <class _arrayT>
  inline _arrayT
  make_array_like(_arrayT const& other)
  {
    std::cerr << "called make_array_like<_arrayT> version\n";
    return _arrayT(other);
  }

  template<class T, std::size_t N>
  inline std::array<T,N>
  make_array_like(std::array<T,N> const& other) noexcept
  {
    std::cerr << "called make_array_like<std::array<T,N>> version\n";
    return std::array<T,N>();
  }

  template <class T>
  inline std::vector<T>
  make_array_like(std::vector<T> const& other)
  {
    std::cerr << "called make_array_like<std::vector<T>> version\n";
    return std::vector<T>(other.size());
  }

  template <class InputIter, class OutputIter, class HMM,
      class transition_matrix = typename HMM::transition_matrix_type,
      class symbols_matrix = typename HMM::symbols_matrix_type,
      class array_type = typename HMM::array_type,
      class float_type = typename HMM::float_type>
  void forward_with_initial(
      InputIter start, InputIter end, OutputIter out, HMM const& hmm,
      array_type const& alpha_0)
  {
    // get HMM properties
    transition_matrix const& A = hmm.transition_matrix();
    symbols_matrix const& B = hmm.symbol_probabilities();
    // do the recursion
    array_type pred_alpha(alpha_0);
    array_type alpha = make_array_like(alpha_0);
    while (!(start == end)) {
      std::size_t ob = *start;
      assert(0 <= ob && ob < hmm.symbols());
      float_type scaling = 0.0;
      for (std::size_t j = 0; j < hmm.states(); ++j) {
        alpha[j] = 0.0;
        for (std::size_t i = 0; i < hmm.states(); ++i)
          alpha[j] += pred_alpha[i]*A[i][j];
        alpha[j] *= B[j][ob];
        scaling += alpha[j];
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

  template <class InputIter, class OutputIter, class HMM,
        class transition_matrix = typename HMM::transition_matrix_type,
        class symbols_matrix = typename HMM::symbols_matrix_type,
        class array_type = typename HMM::array_type,
        class float_type = typename HMM::float_type>
  void
  forward(InputIter start, InputIter end, OutputIter out, HMM const& hmm)
  {
    if (start == end)
      return;
    // get HMM properties
    symbols_matrix const& B = hmm.symbol_probabilities();
    array_type const& pi = hmm.initial_distribution();
    // determine initial alpha_0
    std::size_t ob = *start;
    assert(0 <= ob && ob < hmm.symbols());
    array_type alpha = make_array_like(pi);
    float_type scaling{ 0.0 };
    for (std::size_t i=0; i < hmm.states(); ++i) {
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

    // start the recursion formula with our initial alpha_0
    forward_with_initial(start, end, out, hmm, alpha);
  }


} // namespace hmm
} // namespace nmb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
