#include <fstream>
#include <iostream>
#include <exception>
#include <tuple> // std::tie

#include <boost/iterator/function_input_iterator.hpp>

#include "hidden_markov_model.h"

enum Exit_Error_Codes {
  exit_success = 0,
  exit_not_enough_arguments = 1,
  exit_io_error = 2,
  exit_argument_error = 3
};


std::istream& getline(std::istream& in, std::istringstream& line_stream)
{
  std::string line;
  std::getline(in, line);
  line_stream.str(line);
  line_stream.clear();
  return in;
}

struct model_parse_error: public std::runtime_error {
  model_parse_error(std::string arg): std::runtime_error(arg) {}
};

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence-length>\n";
    return exit_not_enough_arguments;
  }

  std::ifstream input;
  input.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  // Model parameter which will be read in from the model data file.
  std::size_t states;
  std::size_t symbols;
  mnb::hmm::vector::matrix<float> A;
  mnb::hmm::vector::matrix<float> B;
  std::vector<float> pi;

  // Try to read the HMM-model that will be used to generate a random sequence.
  try {
    std::istringstream line;
    input.open(argv[1], std::ifstream::in);
    getline(input, line);
    if (!(line >> states >> symbols))
      throw model_parse_error("Could not read number of states and symbols.");
    A = mnb::hmm::vector::read_matrix<float>(input, states, states);
    B = mnb::hmm::vector::read_matrix<float>(input, states, symbols);
    pi = mnb::hmm::vector::read_array<float>(input, states);
    input.close();
  } catch (std::ifstream::failure& e) {
    std::cerr << "Error while opening or reading from the model data file.\n";
    std::cerr << "Error message is '" << e.what() << "'\n";
    return exit_io_error;
  } catch (model_parse_error& e) {
    std::cerr << "Error while parsing the model data file.\n";
    std::cerr << "Error message is '" << e.what() << "'\n";
    return exit_io_error;
  } catch (mnb::hmm::matrix_read_error& e) {
    std::cerr << "Error while reading matrix or array data.\n";
    std::cerr << "Error message is '" << e.what() << "'\n";
    return exit_io_error;
  }

  std::istringstream obslen_converter(argv[2]);
  std::size_t obslen;
  if (!(obslen_converter >> obslen) || obslen == 0) {
    std::cerr << "Could not convert sequence length to std::size_t.\n";
    return exit_argument_error;
  }

  mnb::hmm::vector::hidden_markov_model<float> hmm(A, B, pi);
  std::function<std::size_t()> generator = mnb::hmm::make_generator(hmm);
  std::cout << obslen << "\n";
  std::copy(boost::make_function_input_iterator(generator, std::size_t{0}),
            boost::make_function_input_iterator(generator, obslen),
            std::ostream_iterator<std::size_t>(std::cout, " "));
  std::cout << std::endl;

  return exit_success;
}
