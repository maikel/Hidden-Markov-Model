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
#else
#define MAIKEL_PROFILER ::maikel::function_profiler __maikel_profiler(__func__, __FILE__)
#endif



namespace maikel {

  /**
   * Tag functions and sum each execution time. Im using this to make a fast
   * and dirty profile of my functions.
   *
   * Example:
   *
   *     void foo_function()
   *     {
   *        function_profiler("foo_function")
   *
   *        // heavy code blah blah ...
   *     }
   *
   *     int main()
   *     {
   *         foo_function();
   *         function_profiler::print_statistics(std::cout);
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
  };


}


#endif /* INCLUDE_FUNCTION_PROFILER_H_ */
