cmake_minimum_required(VERSION 3.10)
project(2dt_match)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_CPP_STANDARD -std=c++14)

pybind11_add_module(rotations rotations.cpp)
