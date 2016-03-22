/*
 * Copyright 2016 Maikel Nadolski
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HMM_ALGORITHM_H_
#define HMM_ALGORITHM_H_

#include <map>
#include <range/v3/core.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/unique.hpp>
#include <gsl_util.h>

#include "maikel/hmm/algorithm/forward.h"
#include "maikel/hmm/algorithm/backward.h"
#include "maikel/hmm/algorithm/baum_welch.h"

namespace maikel {

  template <class SymbolType, class IndexType>
    // requires UnsignedIntegral<I>
    bool is_bijective_index_map(std::map<SymbolType,IndexType> const& map)
    {
      IndexType max_index = ranges::max( map | ranges::view::values );
      std::vector<std::size_t> histogram(max_index+1);
      for (IndexType index : map|ranges::view::values)
        ++histogram[index];
      return ranges::all_of(histogram, [](std::size_t count){ return count == 1; });
    }

  template <class T>
  using ValueType = typename std::remove_reference<T>::type::value_type;

  template <class Index, class Range, class T = ranges::range_value_t<Range>>
    std::map<T,Index>
    map_from_symbols(Range&& range)
    {
      std::map<T,Index> symbols_to_index;
      Index index { 0 };
      ranges::for_each(range | ranges::view::unique, [&symbols_to_index,&index](T const& s){
          symbols_to_index[s] = index++;
      });
      Ensures(is_bijective_index_map(symbols_to_index));
      return symbols_to_index;
    }

}

#endif /* HMM_ALGORITHM_H_ */
