#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>

#include "hidden_markov_model.h"
#include "buffered_binary_iterator.h"

#include <boost/iterator/transform_iterator.hpp>

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
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
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

  // Try to read the HMM-model that will be to do the forward algorithm with
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
  mnb::hmm::vector::hidden_markov_model<float> hmm(A, B, pi);

  // prepare observation stream
  input.clear();
  input.exceptions(std::ifstream::badbit);
  input.open(argv[2], std::ifstream::in);
  std::string line;
  getline(input, line); // remove first line (obs length information)
  auto obs_input_begin = std::istream_iterator<std::size_t>(input);
  auto obs_input_end   = std::istream_iterator<std::size_t>();
  // read observation sequence into vector
  std::vector<std::size_t> obs;
  std::cout << "Reading observation sequence into vector ... " << std::flush;
  auto start = std::chrono::system_clock::now();
  std::copy(obs_input_begin, obs_input_end, std::back_inserter(obs));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> dt = end-start;
  input.close();
  std::cout << "done. Time elapsed: " << dt.count() << "ms.\n";
  std::cout << "size in memory of obs vector: "
      << obs.capacity()*sizeof(decltype(obs)::value_type)/1024/1024 << " MB\n";


  std::vector<std::pair<float, std::vector<float>>> alphas;
  alphas.reserve(obs.size());
  std::cout << "Starting forward algorithm on sequence ... " << std::flush;
  start = std::chrono::system_clock::now();
  hmm.forward(obs.begin(), obs.end(), std::back_inserter(alphas));
  end = std::chrono::system_clock::now();
  dt = end-start;
  std::cout << "done. Time elapsed: " << dt.count() << "ms.\n";

  std::vector<std::vector<float>> betas;
  betas.reserve(obs.size());
  auto first = [](std::pair<float, std::vector<float>> const& p) {
    return p.first;
  };
  auto scaling_begin = boost::make_transform_iterator(alphas.rbegin(), first);
  std::cout << "Starting backward algorithm on sequence ... " << std::flush;
  start = std::chrono::system_clock::now();
  hmm.backward(obs.rbegin(), --obs.rend(), scaling_begin,
      std::back_inserter(betas));
  end = std::chrono::system_clock::now();
  dt = end-start;
  std::cout << "done. Time elapsed: " << dt.count() << "ms.\n";
//  std::cout << "betas.size(): " << betas.size() << std::endl;
//
//  auto beta = betas.rbegin();
//  auto alpha = alphas.begin();
//  std::size_t t = 1;
//  while (alpha != alphas.end() && beta != betas.rend()) {
//    std::cout << "t = " << t << std::endl;
//    std::cout << "scaling: " << alpha->first << "\n";
//    std::cout << "alpha: ";
//    std::copy(alpha->second.begin(), alpha->second.end(), std::ostream_iterator<float>(std::cout, " "));
//    std::cout << "\nbeta: ";
//    std::copy(beta->begin(), beta->end(), std::ostream_iterator<float>(std::cout, " "));
//    std::cout << "\n";
//    ++alpha;
//    ++beta;
//    ++t;
//  }

  return exit_success;
}
