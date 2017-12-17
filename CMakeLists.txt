cmake_minimum_required(VERSION 3.6)
project(mpli)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(mpli ${SOURCE_FILES})

include_directories(/usr/include/mpi/)

set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")