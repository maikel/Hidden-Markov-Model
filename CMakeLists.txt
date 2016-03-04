project(Hidden-Markov-Models)
cmake_minimum_required(VERSION 3.0)

include_directories(include GSL/include)
set(CMAKE_CXX_FLAGS "-std=c++1y -O3 -Wall -DNDEBUG")

add_executable(generate_sequence generate_sequence.cpp)
