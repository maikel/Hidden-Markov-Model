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

#include <cmath>
#include <limits>

#include "../include/types.h"
#include "hidden-markov-models.t.h"

namespace {

CASE ( "test if 0.1 and 0.10000001 are almost equal with ulp = 1" ) {
  float x = 0.1;
  float y = 0.10000001;
  EXPECT(maikel::almost_equal(x,y,1));
}

}
