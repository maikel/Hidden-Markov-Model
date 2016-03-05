cmake_minimum_required( VERSION 2.8 )

project( test )

include_directories  ( "${PROJECT_SOURCE_DIR}/../third_party/eigen"
                       "${PROJECT_SOURCE_DIR}/../third_party/GSL/include"
                       "${PROJECT_SOURCE_DIR}/../third_party/lest/include/lest"
                       "${PROJECT_SOURCE_DIR}/../include" )

set( SOURCES hidden-markov-models.t.cpp arrays.t.cpp arithmetic.t.cpp iodata.t.cpp algorithm.t.cpp )

add_compile_options( -Wall -Wno-missing-braces -std=c++1y )
add_compile_options( -DGSL_THROW_ON_CONTRACT_VIOLATION )
add_executable ( hidden-markov-models.t ${SOURCES} )

enable_testing()

add_test( NAME test COMMAND hidden-markov-models.t --pass )