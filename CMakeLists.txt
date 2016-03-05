project(Hidden-Markov-Models)
cmake_minimum_required(VERSION 3.0)

include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen"
                    "${PROJECT_SOURCE_DIR}/third_party/GSL/include"
                    "${PROJECT_SOURCE_DIR}/include")

add_compile_options( -Wall -O3 -DNDEBUG -std=c++1y )
add_executable(generate_sequence generate_sequence.cpp)
add_executable(forward forward.cpp)
