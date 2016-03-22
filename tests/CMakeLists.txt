cmake_minimum_required( VERSION 2.8 )

project( test )

include_directories  ( "${PROJECT_SOURCE_DIR}/../third_party/eigen"
                       "${PROJECT_SOURCE_DIR}/../third_party/GSL/include"
                       "${PROJECT_SOURCE_DIR}/../third_party/range-v3/include"
                       "${PROJECT_SOURCE_DIR}/../third_party/lest/include/lest"
                       "${PROJECT_SOURCE_DIR}/../include" )

set( SOURCES hidden-markov-models.t.cpp arrays.t.cpp arithmetic.t.cpp iodata.t.cpp algorithm.t.cpp )

add_compile_options( -Wall -Wno-missing-braces -std=c++11 )
add_compile_options( -g -DGSL_THROW_ON_CONTRACT_VIOLATION )
add_executable ( hidden-markov-models.t ${SOURCES} )

add_executable ( function_profiler.t function_profiler.cpp ../include/maikel/function_profiler.cpp )
target_compile_options( function_profiler.t INTERFACE "-O0" )

enable_testing()

add_test( NAME test COMMAND hidden-markov-models.t --pass )
