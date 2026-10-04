cmake_minimum_required(VERSION 3.20)
project(DisaggregatedLLMServingEngine LANGUAGES CXX)

option(DLSE_ENABLE_CUDA "Build DLSE CUDA kernels and benchmarks" OFF)

enable_testing()
add_subdirectory(dlse_runtime)
