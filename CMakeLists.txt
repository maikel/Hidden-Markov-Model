project(Hidden-Markov-Models)
cmake_minimum_required(VERSION 3.0)

SET(CMAKE_BUILD_TYPE Release)

include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen"
                    "${PROJECT_SOURCE_DIR}/third_party/GSL/include"
                    "${PROJECT_SOURCE_DIR}/third_party/range-v3/include"
                    "${PROJECT_SOURCE_DIR}/include")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DGSL_THROW_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -pg -static-libgcc -DGSL_THROW_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")

add_compile_options( -Wall -std=c++11 )
add_executable(generate_sequence generate_sequence.cpp)
add_executable(forward forward.cpp)
