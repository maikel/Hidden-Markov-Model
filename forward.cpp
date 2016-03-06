#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <functional>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/function_output_iterator.hpp>

#include "hmm/hidden_markov_model.h"
#include "hmm/algorithm.h"
#include "hmm/io.h"

using namespace ranges;

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

struct null_output_iterator :
    std::iterator< std::output_iterator_tag,
                   null_output_iterator > {
    /* no-op assignment */
    template<typename T>
    void operator=(T const&) { }

    null_output_iterator & operator++() {
        return *this;
    }

    null_output_iterator operator++(int) {
        return *this;
    }

    null_output_iterator & operator*() { return *this; }
};


int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
    return exit_not_enough_arguments;
  }

  // read model
  std::ifstream model_input(argv[1]);
  auto model = maikel::hmm::read_hidden_markov_model<float>(model_input);

  // prepare reading observation sequence
  using symbol_type = uint8_t;
  using size_type   = std::vector<symbol_type>::size_type;
  std::vector<symbol_type> sequence;

  // read sequence size and reserve memory
  std::ifstream sequence_input(argv[2]);
  size_type length = get_sequence_length<size_type>(sequence_input);
  sequence.reserve(length);
  auto normalize = [] (symbol_type s) { return s - gsl::narrow<symbol_type>('0'); };
  auto sequence_input_range = istream_range<symbol_type>(sequence_input);
  copy(sequence_input_range | view::transform(normalize), back_inserter(sequence));

  // calculate logarithm probability
  float log_probability { 0 };
  auto add_to_logprob = [&log_probability] (float scaling) { log_probability -= std::log(scaling); };
  auto scaling_output_iterator = boost::make_function_output_iterator(add_to_logprob);
  maikel::hmm::forward(model, sequence, null_output_iterator(), scaling_output_iterator);
  std::cout << "log P(O|model) = " << log_probability << std::endl;

  return exit_success;
}
