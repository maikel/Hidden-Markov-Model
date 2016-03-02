#include "array_hmm.h"

#include <fstream>
#include <iostream>

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

  nmb::array_hmm<3, 2> hmm(A, B, pi);
  std::random_device rd;
  nmb::sequence_generator<nmb::array_hmm<3,2>> generator(hmm, rd);
  for (std::size_t k = 0; k < 100; ++k)
    std::cout << generator() << std::endl;

  return 0;
}


