#ifndef ALG_RO_
#define ALG_RO_

#include <algorithm> // all_of
#include <sstream>   // istringstream
#include <iterator>  // iterator_traits
#include <exception> // runtime_error
#include <cmath>     // pow10
#include <istream>
#include <sstream>

#include "../types.h"
#include "gsl_util.h"

#include "hmm/hidden_markov_model.h"

namespace maikel { namespace hmm {

  /**
   * naive implementation of the baum-welch algorithm. just save everything
   * into std::vector's and do not buffer anything in files. Just see if it
   * works.
   */
  template <class float_type, class ObInputIter>
    hidden_markov_model<float_type>
    baum_welch(
        hidden_markov_model<float_type> const& initial_model,
        ObInputIter ob_start, ObInputIter ob_end)
    {
      // typedefs
      using vector_type = typename hidden_markov_model<float_type>::vector_type;
      using matrix_type = typename hidden_markov_model<float_type>::matrix_type;
      using difference_type = typename ObInputIter::difference_type;
      using index_type = typename hidden_markov_model<float_type>::index_type;

      // allocate memory
      std::vector<float> scaling;
      std::vector<vector_type> alphas;
      std::vector<vector_type> betas;
      difference_type sequence_length = std::distance(ob_start, ob_end);
      scaling.reserve(sequence_length);
      alphas.reserve(sequence_length);
      betas.reserve(sequence_length);

      // calculate forward and backward coefficients
      initial_model.forward(ob_start, ob_end, std::back_inserter(alphas), std::back_inserter(scaling));
      auto ob_reverse_start = std::reverse_iterator<ObInputIter>(ob_end);
      auto ob_reverse_end = std::reverse_iterator<ObInputIter>(ob_start);
      initial_model.backward(ob_reverse_start, ob_reverse_end, scaling.rbegin(), std::back_inserter(betas));

      // calculate matrix updates
      matrix_type A = matrix_type::Zero(initial_model.states(), initial_model.states());
      matrix_type B = matrix_type::Zero(initial_model.states(), initial_model.symbols());
      vector_type pi(initial_model.states());

      vector_type gamma_sum = vector_type::Zero(initial_model.states());
      vector_type gamma;
      std::size_t t = 0;
      std::size_t T = gsl::narrow<std::size_t>(sequence_length);

      for (index_type i = 0; i < initial_model.states(); ++i)
        pi(i) = alphas[0](i)*betas[T-1](i) / scaling[0];
      while (ob_start+1 != ob_end && t+1 < T) {
        index_type ob = gsl::narrow<index_type>(*ob_start);
        index_type ob_next = gsl::narrow<index_type>(*(ob_start+1));
        gamma = vector_type::Zero(initial_model.states());
        for (index_type j = 0; j < initial_model.states(); ++j) {
          for (index_type i = 0; i < initial_model.states(); ++i) {
            float entry = alphas[t](i)*initial_model.A(i,j)*initial_model.B(j,ob_next)*betas[T-t-1](j);
            A(i,j) += entry;
            gamma(i) += entry;
          }
        }
        for (index_type j = 0; j < initial_model.states(); ++j) {
          B(j,ob) += gamma(j);
          gamma_sum(j) += gamma(j);
        }
        ++ob_start;
        ++t;
      }
      for (index_type j = 0; j < initial_model.states(); ++j)
        for (index_type i = 0; i < initial_model.states(); ++i)
          A(i,j) /= gamma_sum(i);

      index_type ob = gsl::narrow<index_type>(*ob_start);
      for (index_type i = 0; i < initial_model.states(); ++i) {
        float entry = alphas[T-1](i)*betas[0](i) / scaling[T-1];
        B(i,ob) += entry;
        gamma_sum(i) += entry;
      }

      for (index_type i = 0; i < initial_model.states(); ++i)
        for (index_type k = 0; k < initial_model.symbols(); ++k)
          B(i,k) /= gamma_sum(i);

      return hidden_markov_model<float>(A, B, pi);
    }

} // namespace hmm
} // namespace maikel

#endif /* HIDDEN_MARKOV_MODEL_H_ */
