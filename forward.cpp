#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <cmath>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/log/trivial.hpp>

#include <range/v3/algorithm.hpp>

#include "hmm/hidden_markov_model.h"
#include "hmm/algorithm.h"
#include "hmm/io.h"
#include "function_profiler.h"

using namespace ranges;

enum Exit_Error_Codes {
  exit_success = 0,
  exit_not_enough_arguments = 1,
  exit_io_error = 2,
  exit_argument_error = 3
};

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


inline double my_log(double x) noexcept { return std::log(x); }

int main(int argc, char *argv[])
{

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model.dat> <sequence.dat>\n";
    return exit_not_enough_arguments;
  }
  using float_type = double;

  // read model
  std::ifstream model_input(argv[1]);
  auto model = maikel::hmm::read_hidden_markov_model<float_type>(model_input);

  // prepare reading observation sequence
  using index_type = uint8_t;
//  auto symbols = ranges::view::ints | ranges::view::take(model.symbols());
  std::vector<int> symbols { 0, 1 };
  std::map<int,index_type> symbol_to_index = maikel::map_from_symbols<index_type>(symbols);
  std::ifstream sequence_input(argv[2]);
  BOOST_LOG_TRIVIAL(info) << "Reading sequence ...";
  std::vector<index_type> sequence = maikel::hmm::read_sequence(sequence_input, symbol_to_index);
  BOOST_LOG_TRIVIAL(info) << "Done. Sequence has " << sequence.size() << " symbols.";
//  BOOST_LOG_TRIVIAL(info) << "Starting forward algorithm with storing scaling factors into std::vector.";
//  BOOST_LOG_TRIVIAL(info) << "Transform scaling factors first and then accumulate.";
//  { MAIKEL_PROFILER;
//    std::vector<float_type> scaling;
//    scaling.reserve(sequence.size());
//
//    maikel::hmm::forward(model, sequence, null_output_iterator(), ranges::back_inserter(scaling));
//    { MAIKEL_NAMED_PROFILER("main::transform_first_then_accumulate");
//      ranges::action::transform(scaling, my_log);
//      float_type log_probability = ranges::accumulate(scaling, 0.0);
//      std::cout << "log P(O|model) = " << -log_probability << std::endl;
//    } // MAIKEL_NAMED_PROFILER
//
//  } // MAIKEL_PROFILER
//
//  maikel::function_profiler::print_statistics(std::cerr);
//  maikel::function_profiler::reset();
//
//  BOOST_LOG_TRIVIAL(info) << "Starting forward algorithm with storing scaling factors into std::vector.";
//  BOOST_LOG_TRIVIAL(info) << "Use accumulate on a view::transformed scalingl list.";
//  { MAIKEL_PROFILER;
//    std::vector<float_type> scaling;
//    scaling.reserve(sequence.size());
//
//    maikel::hmm::forward(model, sequence, null_output_iterator(), ranges::back_inserter(scaling));
//    { MAIKEL_NAMED_PROFILER("main::accumulate_viewed_transform");
//        float_type log_probability = ranges::accumulate(scaling | ranges::view::transform(my_log), 0.0);
//        std::cout << "log P(O|model) = " << -log_probability << std::endl;
//      } // MAIKEL_NAMED_PROFILER
//
//    } // MAIKEL_PROFILER
//
//    maikel::function_profiler::print_statistics(std::cerr);
//    maikel::function_profiler::reset();


  BOOST_LOG_TRIVIAL(info) << "Starting forward algorithm with summing scaling factors with an function output iterator.";
  { MAIKEL_PROFILER;
     // calculate logarithm probability
     float_type log_probability = 0.0;
     auto add_to_logprob = [&log_probability] (float_type scaling) {
         log_probability += my_log(scaling);
     };
     auto scaling_output_iterator = boost::make_function_output_iterator(add_to_logprob);
     maikel::hmm::forward(model, sequence, null_output_iterator(), scaling_output_iterator);
     std::cout << "log P(O|model) = " << -log_probability << std::endl;
  }

  maikel::function_profiler::print_statistics(std::cerr);

  return exit_success;
}
