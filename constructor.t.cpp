#include "array_hmm.h"

#include <fstream>
#include <iostream>

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

  std::cout << "pi is probability array: " <<
      nmb::is_probability_array(pi) << std::endl;

  std::ifstream matrix_file("A.dat");
  matrix_file >> A;
  matrix_file.close();
  matrix_file.open("B.dat");
  matrix_file >> B;
  matrix_file.close();

  const std::size_t ob_len = 20000;

  nmb::array_hmm<3, 2> hmm(A, B, pi);
  std::random_device rd;
  nmb::sequence_generator<nmb::array_hmm<3,2>> generator(hmm, rd);
  std::vector<std::size_t> observation;
  for (std::size_t k = 0; k < ob_len; ++k)
    observation.push_back(generator());

  std::vector<std::pair<float, std::array<float,3>>> alphas;
  nmb::forward(observation.begin(), observation.end(),
      std::back_inserter(alphas), hmm);
  for (const auto& p = alphas.rbegin(); p != alphas.rbegin()+20; ++p) {
    auto &alpha = p.second;
    std::cout << p.first << ": ";
    std::copy(alpha.begin(), alpha.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";
  }

  return 0;
}


