cmake_minimum_required(VERSION 3.20)
project(DLSE LANGUAGES CXX)

option(DLSE_ENABLE_CUDA "Build CUDA benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dlse_runtime_core
    src/runtime.cpp
)

target_include_directories(
    dlse_runtime_core
    PUBLIC include
)

target_compile_options(
    dlse_runtime_core
    PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(
    dlse_runtime_tests
    tests/runtime_tests.cpp
)

target_link_libraries(
    dlse_runtime_tests
    PRIVATE dlse_runtime_core
)

enable_testing()
add_test(
    NAME dlse_runtime_tests
    COMMAND dlse_runtime_tests
)

add_executable(
    dlse_runtime_bench
    benchmarks/runtime_bench.cpp
)

target_link_libraries(
    dlse_runtime_bench
    PRIVATE dlse_runtime_core
)

if(DLSE_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(
        dlse_cuda
        STATIC
        cuda/paged_attention.cu
    )

    target_include_directories(
        dlse_cuda
        PUBLIC include
    )

    target_link_libraries(
        dlse_cuda
        PUBLIC CUDA::cudart
    )

    set_target_properties(
        dlse_cuda
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
    )

    add_executable(
        dlse_paged_attention_bench
        benchmarks/paged_attention_bench.cu
    )

    target_link_libraries(
        dlse_paged_attention_bench
        PRIVATE dlse_cuda CUDA::cudart
    )

    add_executable(
        dlse_cuda_graph_bench
        benchmarks/cuda_graph_bench.cu
    )

    target_link_libraries(
        dlse_cuda_graph_bench
        PRIVATE CUDA::cudart
    )
endif()
