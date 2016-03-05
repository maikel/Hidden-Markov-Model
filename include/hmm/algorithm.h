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


namespace maikel { namespace hmm {

  template<class float_type,
            typename index_type = typename VectorX<float_type>::Index>
    index_type
    find_by_distribution(const VectorX<float_type>& dist, float_type X) noexcept
    {
      float_type P_fn = 0;
      index_type state = 0;
      index_type max = dist.size();
      while (state < max) {
        P_fn += dist(state);
        if (P_fn < X)
          ++state;
        else
          break;
      }
      return state;
    }

  template<class float_type,
            class index_type = typename MatrixX<float_type>::Index>
    index_type
    find_by_distribution(const MatrixX<float_type>& dist, index_type row, float_type X) noexcept
    {
      Expects(row < dist.rows());
      float_type P_fn = 0;
      index_type state = 0;
      index_type cols = dist.cols();
      while (state < cols) {
        P_fn += dist(row, state);
        if (P_fn < X)
          ++state;
        else
          break;
      }
      return state;
    }


} // namespace hmm
} // namespace nmb

#endif /* HIDDEN_MARKOV_MODEL_H_ */
