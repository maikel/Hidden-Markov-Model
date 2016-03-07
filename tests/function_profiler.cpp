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

#include <iostream>
#include <limits>
#include "function_profiler.h"


void barfoo()
{
  maikel::function_profiler(__PRETTY_FUNCTION__);
  std::size_t max = 200000000;
  std::size_t n = 0;
  for (std::size_t i = 0; i < max; ++i) n += i;
}

void foobar()
{
  maikel::function_profiler(__PRETTY_FUNCTION__);
  int max = 20000;
  int n = 0;
  for (int i = 0; i < max; ++i) n += i*i;
}

int main()
{
  barfoo();
  barfoo();
  for (int i =0; i < 10; ++i) foobar();
  maikel::function_profiler::print_statistics(std::cout);
}
