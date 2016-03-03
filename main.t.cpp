#include "algorithms.h"
#include "array_hmm.h"

#include <fstream>
#include <iostream>

#include <chrono>

#include <boost/iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

std::ostream& operator<<(std::ostream& out, std::array<float, 3> const& alpha)
{
  copy(alpha.begin(), alpha.end(), std::ostream_iterator<float>(out, " "));
  return out;
}

int main(int argc, char *argv[])
{
  nmb::matrix_type<float, 3, 3> A;
  nmb::matrix_type<float, 3, 2> B;
  std::array<float, 3> pi {0.4, 0.4, 0.2};

  std::ifstream matrix_file("A.dat");
  matrix_file >> A;
  matrix_file.close();
  matrix_file.open("B.dat");
  matrix_file >> B;
  matrix_file.close();

  const std::size_t ob_len = 10;

  nmb::array_hmm<3, 2> hmm(A, B, pi);
  std::random_device rd;
  nmb::sequence_generator<nmb::array_hmm<3,2>> generator(hmm, rd);
  std::vector<std::size_t> obs;
  for (std::size_t k = 0; k < ob_len; ++k)
    obs.push_back(generator());

  std::vector<std::pair<float, std::array<float,3>>> alphas;
  std::vector<std::array<float, 3>> betas;

  std::cout << "start" << std::endl;
  auto start = std::chrono::system_clock::now();
//  nmb::forward(obs.begin(), obs.end(), std::back_inserter(alphas), hmm);
//  auto scaling_start = boost::make_transform_iterator(alphas.rbegin(),
//      [](auto const& pair) { return pair.first; });
//  nmb::backward(obs.begin(), obs.end(), scaling_start,
//      std::back_inserter(betas), hmm);
  auto hmm_approx = nmb::baum_welch(obs.begin(), obs.end(), hmm);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  std::cout << "end time: " << diff.count() << " milliseconds." << std::endl;

  return 0;
}


