#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <cmath>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/function_input_iterator.hpp>
#include <boost/log/trivial.hpp>

#include <range/v3/algorithm.hpp>

#include "maikel/hmm/hidden_markov_model.h"
#include "maikel/hmm/algorithm.h"
#include "maikel/hmm/io.h"
#include "maikel/function_profiler.h"


enum Exit_Error_Codes {
  exit_success = 0,
  exit_not_enough_arguments = 1,
  exit_io_error = 2,
  exit_argument_error = 3
};

template <class float_type, class index_type>
  void accumulate_and_view_transform(
      std::vector<index_type> const& sequence,
      maikel::hmm::hidden_markov_model<float_type> const& model)
  {
    BOOST_LOG_TRIVIAL(info) << "Starting forward algorithm with storing scaling factors into std::vector.";
    BOOST_LOG_TRIVIAL(info) << "Use accumulate on a view::transformed scaling list.";
    MAIKEL_PROFILER;

    float_type logprob = 0.0;
    for (auto&& scaled_alpha : maikel::hmm::forward(sequence, model))
      logprob += std::log(scaled_alpha.first);
    std::cout << -logprob << std::endl;
  }

int main(int argc, char *argv[])
{

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
    return exit_not_enough_arguments;
  }
  using float_type = double;
  using index_type = uint8_t;

  // read model
  std::ifstream model_input(argv[1]);
  auto model = maikel::hmm::read_hidden_markov_model<float_type>(model_input);

  std::vector<int> symbols { 0, 1 };
  std::map<int,index_type> symbol_to_index = maikel::map_from_symbols<index_type>(symbols);
  std::ifstream sequence_input(argv[2]);
  BOOST_LOG_TRIVIAL(info) << "Reading sequence ...";
  std::vector<index_type> sequence = maikel::hmm::read_sequence(sequence_input, symbol_to_index);
  BOOST_LOG_TRIVIAL(info) << "Done. Sequence length is " << sequence.size() << " ("
      << sequence.size() / sizeof(index_type) / 1024 / 1024 << " mega bytes)";
  maikel::function_profiler::print_statistics(std::cerr);

  maikel::function_profiler::reset();
  accumulate_and_view_transform(sequence, model);
  maikel::function_profiler::print_statistics(std::cerr);

  return exit_success;
}
