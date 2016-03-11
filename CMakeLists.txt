project(Hidden-Markov-Models)
cmake_minimum_required(VERSION 3.0)

SET(CMAKE_BUILD_TYPE Profile)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen"
                    "${PROJECT_SOURCE_DIR}/third_party/GSL/include"
                    "${PROJECT_SOURCE_DIR}/third_party/range-v3/include"
                    "${PROJECT_SOURCE_DIR}/include")
                    
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DGSL_THROW_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -DMAIKEL_PROFILE_FUNCTIONS -DNDEBUG -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")

add_compile_options( -Wall -Wpedantic -std=c++11 )
add_executable(generate_sequence generate_sequence.cpp)

add_executable(forward forward.cpp include/maikel/function_profiler.cpp)
target_link_libraries(forward boost_log pthread)
target_compile_options(forward PUBLIC -DBOOST_LOG_DYN_LINK)