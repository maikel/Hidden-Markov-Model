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

#ifndef INCLUDE_MAIKEL_ITERATOR_GETLINES_H_
#define INCLUDE_MAIKEL_ITERATOR_GETLINES_H_

#include <istream>
#include <string>

#include <boost/iterator/iterator_facade.hpp>

#include <gsl_assert.h>

namespace maikel {

  class getlines {

    private:
      std::istream& sin_; // not owning
      std::string line_;

    public:
      virtual ~getlines() = default;

      explicit getlines(std::istream& sin)
      : sin_{sin}
      {
        next();
      }

      class iterator
          : public boost::iterator_facade<iterator, std::string, std::input_iterator_tag>
      {
        private:
          getlines* rng_ = nullptr;

        public:
          iterator() = default;

        private:
          friend class getlines;
          friend class boost::iterator_core_access;

          iterator(getlines& rng)
          : rng_{ rng ? &rng : nullptr } {}

          inline std::string& dereference() const
          {
            Expects(rng_);
            return rng_->line_;
          }

          inline bool equal(iterator other) const noexcept
          {
            return rng_ == other.rng_;
          }

          inline void increment()
          {
            Expects(rng_);
            if (!rng_->next())
              rng_ = nullptr;
          }
      };

      inline operator bool()
      {
        return sin_;
      }

      inline bool next()
      {
        return std::getline(sin_, line_);
      }

      inline iterator begin() noexcept
      {
        return {*this};
      }

      inline iterator end() noexcept
      {
        return {};
      }
  };




}

#endif /* INCLUDE_MAIKEL_ITERATOR_GETLINES_H_ */
