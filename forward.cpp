#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>

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
            << sequence.capacity()*sizeof(symbol_type)/1024/1024 << " mega bytes.";
  std::cout << std::endl;

//  std::copy(sequence.begin(), sequence.end(), std::ostream_iterator<uint8_t>(std::cout, " "));
//  std::cout << std::endl;

  return exit_success;
}
