project(Hidden-Markov-Models)

include_directories(include GSL/include)
set(CMAKE_CXX_FLAGS "-std=c++1y -O3 -Wall")

add_executable(my_tests main.t.cpp)