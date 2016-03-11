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

#ifndef INCLUDE_FUNCTION_PROFILER_H_
#define INCLUDE_FUNCTION_PROFILER_H_

#include <chrono>
#include <map>

#ifndef MAIKEL_PROFILE_FUNCTIONS
#define MAIKEL_PROFILER
#define MAIKEL_NAMED_PROFILER
#else
#define MAIKEL_PROFILER ::maikel::function_profiler __maikel_function_profiler(__func__, __FILE__)
#define MAIKEL_NAMED_PROFILER(x) ::maikel::function_profiler __maikel_function_profiler(x, __FILE__)
#endif



namespace maikel {

  struct timer_is_currently_active: public std::runtime_error {
      timer_is_currently_active(): std::runtime_error(
          "Can not reset time table because there is currently a running profiler.") {}
  };

  /**
   * Tag functions and sum times for each execution. Im using this to make fast
   * and dirty profiling for my functions.
   *
   * Example:
   *
   *     void foo_function()
   *     {
   *        maikel::function_profiler profiler("name","group");
   *
   *        // heavy code blah blah ...
   *     }
   *
   *     int main()
   *     {
   *         foo_function();
   *         maikel::function_profiler::print_statistics(std::cout);
   *     }
   */
  class function_profiler {
    public:
      using clock = std::chrono::high_resolution_clock;

    private:
      static std::map<std::string, clock::duration> time_table;
      static bool is_active;
      static clock::duration total_duration;

      decltype(time_table)::iterator function_iterator;
      clock::time_point start = clock::now();
      bool is_top_level = false;

    public:
      function_profiler(std::string function_name, std::string file_name) noexcept;
      ~function_profiler() noexcept;
      static void print_statistics(std::ostream& out);
      static void reset();
  };


}


#endif /* INCLUDE_FUNCTION_PROFILER_H_ */
