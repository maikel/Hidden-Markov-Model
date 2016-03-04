#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>

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
  auto obs_input_begin = std::istream_iterator<uint8_t>(input);
  auto obs_input_end   = std::istream_iterator<uint8_t>();
  // read observation sequence into vector
//  std::vector<uint8_t> obs;
//  std::cout << "Reading observation sequence into vector ... " << std::flush;
//  auto start = std::chrono::system_clock::now();
//  std::copy(obs_input_begin, obs_input_end, std::back_inserter(obs));
//  auto end = std::chrono::system_clock::now();
//  std::chrono::duration<float, std::milli> dt = end-start;
//  std::cout << "done. Time elpased: " << dt.count() << "ms.\n";
//  input.close();

//  std::cout << "size in memory of obs vector: "
//      << obs.capacity()*sizeof(decltype(obs)::value_type)/1024/1024 << " MB\n";

  // prepare alpha output stream
  std::ofstream alpha_file("alpha.bin.dat", std::ofstream::binary | std::ofstream::out);

  std::cout << "Starting forward algorithm on sequence ... " << std::flush;
  auto start = std::chrono::system_clock::now();
//  hmm.forward(obs_input_begin, obs_input_end,
//      std::ostreambuf_iterator<char>(alpha_file));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> dt = end-start;
  std::cout << "done. Time elpased: " << dt.count() << "ms.\n";
  alpha_file.close();


  return exit_success;
}
