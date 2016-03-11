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


#include "maikel/function_profiler.h"

#include <range/v3/algorithm/max.hpp>
#include <range/v3/action/insert.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/reverse.hpp>
#include <iomanip>
#include <utility>

namespace maikel {

  std::map<std::string, function_profiler::clock::duration> function_profiler::time_table;
  bool function_profiler::is_active = false;
  function_profiler::clock::duration function_profiler::total_duration = function_profiler::clock::duration::zero();


  function_profiler::function_profiler(std::string function_name, std::string file_name) noexcept
  : is_top_level(!is_active)
  {
    is_active = true;
    std::string func_id(function_name + "::" + file_name);
    std::tie(function_iterator, std::ignore) =
        time_table.insert(std::make_pair(func_id, clock::duration::zero()));
  }

  function_profiler::~function_profiler() noexcept
  {
    auto diff = clock::now()-start;
    function_iterator->second += diff;
    if (is_top_level) {
      total_duration += diff;
      is_active = false;
    }
  }

  void function_profiler::print_statistics(std::ostream& out)
  {
    if (time_table.empty())
      return;

    std::multimap<clock::duration, std::string> functions_sorted_by_duration;
    ranges::insert(functions_sorted_by_duration,
            ranges::view::zip( time_table | ranges::view::values,
                               time_table | ranges::view::keys    ));
    std::size_t max_name_length =
        ranges::max(time_table | ranges::view::keys
                               | ranges::view::transform(ranges::size));
    std::string name;
    clock::duration time;
    auto total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total_duration);
    out << "Total traced execution time: " << total_ms.count() << "ms.\n";
    out << "Printing time table of traced functions:\n";
    for (auto const& function_info : ranges::view::reverse(functions_sorted_by_duration)) {
      std::tie(time, name) = function_info;
      out << std::setw(10) << time.count()*100/total_duration.count() << "% ";
      out << std::setw(10);
      auto time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time);
      out << time_ms.count() << "ms ";
      out << std::setw(max_name_length + 2) << name << "\n";
    }
  }

  void function_profiler::reset()
  {
    if (is_active)
      throw timer_is_currently_active();

    time_table.clear();
    total_duration = clock::duration::zero();
  }

} // namespace maikel
