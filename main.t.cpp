#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include "hidden_markov_model.h"

#include <boost/iterator/function_input_iterator.hpp>

std::ostream& operator<<(std::ostream& out, std::array<float, 3> const& alpha)
{
  copy(alpha.begin(), alpha.end(), std::ostream_iterator<float>(out, " "));
  return out;
}

int main(int argc, char *argv[])
{
  mnb::hmm::matrix<float,3,3> A;
  mnb::hmm::matrix<float,3,2> B;
  std::array<float,3> pi { 0.4, 0.4, 0.2 };
  std::ifstream A_in("A.dat");
  A_in >> A;
  std::ifstream B_in("B.dat");
  B_in >> B;
  mnb::hmm::hidden_markov_model<float,3,2> hmm(A, B, pi);
  std::random_device rd;
  auto generator = mnb::hmm::make_generator(hmm);
  auto gen_start = boost::make_function_input_generator(generator, 0);
  auto gen_end = boost::make_function_input_generator(generator, 100);
  std::copy(gen_start, gen_end,
      std::ostream_iterator<std::size_t>(std::cout, " "));
  std::cout << hmm.states() << std::endl;
  return 0;
}


