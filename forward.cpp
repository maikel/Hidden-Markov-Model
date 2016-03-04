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
//  alphas.reserve(obs.size());

  // prepare alpha output stream
  std::fstream alpha_file("alpha.bin.dat",
      std::fstream::binary | std::fstream::out);
  std::cout << "Starting forward algorithm on sequence ... " << std::flush;
  start = std::chrono::system_clock::now();
  hmm.forward(obs.begin(), obs.end(),
      mnb::ostream_buffered_binary_iterator<float, 1>(alpha_file));
//      std::back_inserter(alphas));
  alpha_file.close();
  end = std::chrono::system_clock::now();
  dt = end-start;
  std::cout << "done. Time elapsed: " << dt.count() << "ms.\n";

  alpha_file.clear();
  alpha_file.open("alpha.bin.dat", std::fstream::binary | std::fstream::in);
  auto alphas_begin = mnb::alphas_binary_input_iterator<float>(alpha_file, states);
  auto alphas_end = mnb::alphas_binary_input_iterator<float>();
//  auto alphas_begin = alphas.begin();
//  auto alphas_end = alphas.end();
  auto logfirst = [](auto const& p) {
    return -std::log(p.first);
  };
  auto scaling_begin = boost::make_transform_iterator(alphas_begin, logfirst);
  auto scaling_end = boost::make_transform_iterator(alphas_end, logfirst);
  std::cout << "Calculate log Probability from binary file ... " << std::flush;
//  std::cout << "Calculate log Probability from array ... " << std::flush;
  start = std::chrono::system_clock::now();
  float P = std::accumulate(scaling_begin, scaling_end, 0.0f);
  end = std::chrono::system_clock::now();
  dt = end-start;
  std::cout << "done. Time elapsed: " << dt.count() << "ms.\n";
  std::cout << "log P(O | lambda) = " << P << std::endl;

//  for (auto const& p : alphas) {
//    std::cout << "scaling: " << p.first << ", alpha: ";
//    std::copy(p.second.begin(), p.second.end(),
//        std::ostream_iterator<float>(std::cout, " "));
//    std::cout << "\n";
//  }

  return exit_success;
}
