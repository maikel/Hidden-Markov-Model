#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <functional>

#include <boost/iterator/transform_iterator.hpp>

#include "hmm/hidden_markov_model.h"
#include "hmm/iodata.h"

enum Exit_Error_Codes {
  exit_success = 0,
  exit_not_enough_arguments = 1,
  exit_io_error = 2,
  exit_argument_error = 3
};

template <class size_type>
  size_type get_sequence_length(std::istream& in)
  {
    std::string line;
    std::getline(in, line);
    std::istringstream linestream(line);
    size_type length;
    linestream >> length;
    return length;
  }

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
    return exit_not_enough_arguments;
  }
  // prepare some timer
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  std::chrono::duration<double, std::milli> dt;

  // read model
  std::ifstream model_input("model.dat");
  auto model = maikel::hmm::read_hidden_markov_model<float>(model_input);

  // prepare reading observation sequence
  using symbol_type = uint8_t;
  using size_type   = std::vector<symbol_type>::size_type;
  std::vector<symbol_type> sequence;

  // read sequence size and reserve memory
  std::ifstream sequence_input("sequence.dat");
  size_type length = get_sequence_length<size_type>(sequence_input);
  sequence.reserve(length);

  auto normalize = [] (symbol_type s) { return s - gsl::narrow<symbol_type>('0'); };
  auto sequence_input_begin = boost::make_transform_iterator(
      std::istream_iterator<symbol_type>(sequence_input), normalize);
  auto sequence_input_end = boost::make_transform_iterator(
      std::istream_iterator<symbol_type>(), normalize);

  std::cout << "Read observation data ... " << std::flush;
  start = std::chrono::system_clock::now();
  // read seqeunce data from file into std::vector
  std::copy(sequence_input_begin, sequence_input_end, std::back_inserter(sequence));
  end = std::chrono::system_clock::now();
  dt = end-start;
  std::cout << " time duration: " << dt.count() << "ms.\n";
  std::cout << "Sequence size in memory: "
            << sequence.capacity()*sizeof(symbol_type) << " bytes.\n";

  // prepare alpha vector
  using alpha_type = decltype(model)::vector_type;
  std::vector<float> scaling;
  scaling.reserve(sequence.size());
  std::vector<alpha_type> alphas;
  alphas.reserve(sequence.size());
  std::cout << "Starting forward algorithm ... " << std::flush;
  start = std::chrono::system_clock::now();
  model.forward(sequence.begin(), sequence.end(), std::back_inserter(alphas), std::back_inserter(scaling));
  end = std::chrono::system_clock::now();
  dt = end-start;
  std::cout << " time duration: " << dt.count() << "ms.\n";

  auto logarithm = [](float x) { return std::log(x); };
  float P = std::accumulate(
      boost::make_transform_iterator(scaling.begin(), logarithm),
      boost::make_transform_iterator(scaling.end(), logarithm), 0.0f);
  std::cout << -P << std::endl;

  std::cout << "alpha (last 10):\n";
  std::copy(alphas.rbegin(), alphas.rbegin()+10, std::ostream_iterator<alpha_type>(std::cout,"\n"));

  return exit_success;
}
