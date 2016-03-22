#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <cmath>
#include <iterator>

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
  void accumulate_scaling_and_write_alpha_to_file(
      std::vector<index_type> const& sequence,
      maikel::hmm::hidden_markov_model<float_type> const& model)
  {
    std::size_t states = model.states();
//    std::vector<Eigen::RowVectorXd> alphas(T, Eigen::RowVectorXd(states));
    std::ofstream alphas("alphas.bin", std::ofstream::binary);
    BOOST_LOG_TRIVIAL(info) << "Starting forward algorithm with storing scaling factors into std::vector.";
    BOOST_LOG_TRIVIAL(info) << "Use accumulate on a view::transformed scaling list.";
    float_type logprob = 0.0;
    std::size_t datalen = sizeof(float_type)*states;
    { MAIKEL_PROFILER;
      std::ostreambuf_iterator<char> out(alphas);
      for (auto&& scaled_alpha : maikel::hmm::forward(sequence, model)) {
        logprob += std::log(scaled_alpha.first);
        std::copy_n(reinterpret_cast<const char*>(scaled_alpha.second.data()), datalen, out);
      }
    }
    std::cout << -logprob << std::endl;
  }

template <class T>
void read_alphas_from_bin(const maikel::hmm::hidden_markov_model<T>& hmm)
{
  MAIKEL_PROFILER;
  std::ifstream alphas("alphas.bin", std::ifstream::binary);
  Eigen::Matrix<T, 1, Eigen::Dynamic> alpha(hmm.states());
  std::size_t data_len = sizeof(T)*alpha.size();
  std::istreambuf_iterator<char> in(alphas), end;
  while (in != end) {
    std::copy_n(in, data_len, reinterpret_cast<char*>(alpha.data()));
    std::advance(in, data_len+1);
  }
}

int main(int argc, char *argv[])
{
  using namespace std;
  using namespace maikel::hmm;

  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
    return exit_not_enough_arguments;
  }
  using float_type = double;
  using index_type = uint8_t;

  // read model
  ifstream model_input(argv[1]);
  auto model = read_hidden_markov_model<float_type>(model_input);

  vector<int> symbols { 0,1 };
  map<int,index_type> symbol_to_index = maikel::map_from_symbols<index_type>(symbols);
  ifstream sequence_input(argv[2]);
  vector<index_type> sequence = read_sequence(sequence_input, symbol_to_index);

  {
    MAIKEL_NAMED_PROFILER("v2::forward");
    float_type scaling = 0;
    for (auto&& alpha : forward(begin(sequence), end(sequence), model)) {
      scaling += log(alpha.first);
    }
    cout << -scaling << endl;
  }
  maikel::function_profiler::print_statistics(cout);

  return exit_success;
}
